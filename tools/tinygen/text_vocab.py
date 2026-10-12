"""Byte-level BPE vocabulary for the tiny random-weight GGUFs, so that a llama-server on them accepts TEXT.

The default vocabulary of tools/tinygen/*_tiny.py is a minimal SPM list ("t3", "t4", ...): prompts must be token-id
lists, and any text is refused (llama.cpp @79e2e74: POST /tokenize with text answers HTTP 500 "unordered_map::at":
SPM falls back to byte tokens "<0xNN>", then to the byte as a one-character string, and std::unordered_map::at throws
when neither is in the list, src/llama-vocab.cpp:177-196, 4149-4162). Tools that send text - the bench
harness (/tokenize prompt fitting, chat and completion requests), the gateway's context guard, logprob_compare -
cannot run against such a model.

--text-vocab replaces it with a GPT-2 style byte-level BPE vocabulary (tokenizer model "gpt2", pre-tokenizer
"default"), the family the target models use (Qwen and GLM GGUFs are byte-level BPE):
  ids 0, 1, 2   <unk>, <s> (BOS), </s> (EOS)  - the same ids as the default vocabulary
  ids 3..258    the 256 single bytes, in byte order, as GPT-2 byte-to-unicode characters
  ids 259..     a few merges ("Ġ" + a lowercase letter, and "ĊĊ"), so that tokens != bytes and prompt fitting
                by /tokenize is not a plain character count
Every byte sequence tokenizes and detokenizes. The weights stay random: generated text is noise, so this model can
exercise formats and protocols, never quality. No chat template is written: llama-server then uses its built-in
chatml (common/chat.cpp CHATML_TEMPLATE_SRC @79e2e74).
"""


def bytes_to_unicode():
    """GPT-2's reversible byte -> printable unicode character map (openai/gpt-2 encoder.py)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def byte_bpe_vocab():
    """Returns (tokens, token_types, merges). Types: 1 normal, 2 unknown, 3 control (gguf TokenType)."""
    b2u = bytes_to_unicode()
    tokens = ["<unk>", "<s>", "</s>"] + [b2u[b] for b in range(256)]
    space, newline = b2u[ord(" ")], b2u[ord("\n")]
    merges = ["%s %s" % (space, chr(c)) for c in range(ord("a"), ord("z") + 1)] + ["%s %s" % (newline, newline)]
    tokens += [m.replace(" ", "") for m in merges]
    types = [2, 3, 3] + [1] * (len(tokens) - 3)
    assert len(set(tokens)) == len(tokens)
    return tokens, types, merges


N_TEXT_VOCAB = len(byte_bpe_vocab()[0])


def write_byte_bpe_vocab(w, bos=1, eos=2):
    tokens, types, merges = byte_bpe_vocab()
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_bos_token_id(bos)
    w.add_eos_token_id(eos)
    w.add_unk_token_id(0)
    w.add_add_bos_token(False)
    return len(tokens)
