FILE: slug.py
```python
"""URL slugs (SYNTHETIC fixture)."""
import re
import unicodedata


def slugify(text):
    """Return a URL slug of text (see the task for the exact rules)."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
```
