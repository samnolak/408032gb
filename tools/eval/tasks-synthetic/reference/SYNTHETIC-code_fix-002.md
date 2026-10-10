The token pattern does not accept the hour unit.

FILE: durations.py
```python
"""Duration parsing (SYNTHETIC fixture)."""
import re

_UNITS = {"h": 3600, "m": 60, "s": 1}
_TOKEN = re.compile(r"(\d+)([hms])")


def parse_duration(text):
    """Parse '1h30m', '45s', '2m5s' into seconds. Raises ValueError on anything else."""
    text = text.strip().lower()
    if not text:
        raise ValueError("empty duration")
    pos, total = 0, 0
    for m in _TOKEN.finditer(text):
        if m.start() != pos:
            raise ValueError("bad duration: %r" % text)
        total += int(m.group(1)) * _UNITS[m.group(2)]
        pos = m.end()
    if pos != len(text):
        raise ValueError("bad duration: %r" % text)
    return total
```
