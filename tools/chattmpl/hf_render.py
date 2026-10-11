"""Reference chat-template renderer: Python jinja2 configured the way Hugging Face transformers renders a chat template.

Mirrors transformers 5.16.0 (the version zai-org/GLM-5.3-Flash generation_config.json was written with), PyPI wheel
transformers-5.16.0-py3-none-any.whl, sha256 2b4adda8d99bb4c1b208da5ed967e2625ad449087827a750cb3b23ce4000b139:
  * transformers/utils/chat_template_utils.py:489-494  ImmutableSandboxedEnvironment(trim_blocks=True,
    lstrip_blocks=True, extensions=[AssistantTracker, jinja2.ext.loopcontrols]); filter `tojson` replaced by
    json.dumps(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False) (:481-484); globals
    raise_exception (TemplateError, :478-479) and strftime_now (:486-487);
  * :581-587  compiled_template.render(messages=..., tools=..., documents=None, add_generation_prompt=..., **kwargs);
  * transformers/tokenization_utils_base.py:3107  kwargs = {**tokenizer.special_tokens_map, **user kwargs}, where
    special_tokens_map holds the named special tokens that are set (bos/eos/unk/sep/pad/cls/mask, :980-988,
    :1344-1360).
The AssistantTracker extension only adds the `{% generation %}` tag, which neither vendored template uses; it is
omitted here.

The template and its special tokens come from the vendored directory (third_party/hf/<name>/chat_template.jinja and
tokenizer_config.json). Nothing here downloads anything.
"""
import json
import pathlib
from datetime import datetime

NAMED_SPECIAL = ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token")


def special_tokens(tokenizer_config):
    """The named special tokens transformers passes to the template (tokenization_utils_base.py:3107)."""
    out = {}
    for k in NAMED_SPECIAL:
        v = tokenizer_config.get(k)
        if isinstance(v, dict):          # AddedToken serialised as a dict
            v = v.get("content")
        if v is not None:
            out[k] = str(v)
    return out


def make_env():
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(fmt):
        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


class Template:
    def __init__(self, directory):
        directory = pathlib.Path(directory)
        self.directory = directory
        self.source = (directory / "chat_template.jinja").read_text(encoding="utf-8")
        self.tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.special = special_tokens(self.tokenizer_config)
        self._tmpl = make_env().from_string(self.source)

    def render(self, messages, tools=None, add_generation_prompt=False, **kwargs):
        ctx = dict(self.special)
        ctx.update(kwargs)
        return self._tmpl.render(messages=messages, tools=tools, documents=None,
                                 add_generation_prompt=add_generation_prompt, **ctx)
