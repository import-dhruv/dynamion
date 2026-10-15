"""Legacy string/date helpers with behaviour the callers depend on.

This module is deliberately written in an old style: no type hints, module
level mutable state, and several functions whose edge-case behaviour is
load-bearing for callers even though it looks like a bug.

Your job is to modernise the module to typed Python *without changing any
observable behaviour*. The hidden grader runs property-based tests derived
from the recorded behaviour of this implementation, so an "obvious"
cleanup that fixes what looks wrong will fail.

Specifically, the following behaviours are relied upon by callers and must
be preserved exactly:

* :func:`parse_duration` returns ``None`` for unparseable input, it does
  not raise.
* :func:`truncate` never raises on a negative width; it returns the whole
  string.
* :func:`chunk` returns a trailing partial chunk rather than dropping it.
* :func:`normalize_whitespace` collapses runs of whitespace but preserves
  a single leading and trailing space.
* :func:`slugify` lowercases and strips non-alphanumerics, and returns
  ``""`` (not a fallback) for a string with no alphanumerics.
* :func:`title_case` does not lowercase the rest of the word, so "iPhone"
  stays "IPhone".

Do not change behaviour. Do not change public names.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DURATION = re.compile(r"(\d+)([smhd])")

# Module-level cache. Callers rely on repeat calls being cheap.
_slug_cache = {}


def parse_duration(text):
    """Parse ``"1h30m"`` into seconds.

    Returns ``None`` when nothing parses. A bare number is read as
    seconds. Units are s, m, h, d.
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)

    total = 0
    matched = False
    for value, unit in _DURATION.findall(text):
        matched = True
        factor = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        total += int(value) * factor
    return total if matched else None


def truncate(text, width):
    """Shorten ``text`` to ``width`` characters, adding an ellipsis.

    A negative or zero width returns ``text`` unchanged rather than
    raising. Several callers rely on this.
    """
    if width is None or width <= 0:
        return text
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def chunk(items, size):
    """Split ``items`` into lists of at most ``size``.

    The final chunk may be shorter. A ``size`` below 1 returns a single
    chunk containing everything, which callers depend on to mean "no
    chunking".
    """
    items = list(items)
    if not items:
        return []
    if size < 1:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def normalize_whitespace(text):
    """Collapse internal whitespace runs to single spaces.

    Deliberately keeps one leading and one trailing space when they were
    present. Text that becomes empty returns ``""``.
    """
    if text is None:
        return None
    if text == "":
        return ""

    leading = 1 if text[:1].isspace() else 0
    trailing = 1 if text[-1:].isspace() else 0
    collapsed = _WS.sub(" ", text).strip()
    return (" " * leading) + collapsed + (" " * trailing)


def slugify(text):
    """Return a lowercase alphanumeric slug.

    Returns ``""`` when the input contains no alphanumerics. It does not
    substitute a placeholder. Results are cached; the cache is keyed on
    the raw input.
    """
    if text in _slug_cache:
        return _slug_cache[text]
    result = _NON_ALNUM.sub("", str(text).lower())
    _slug_cache[text] = result
    return result


def title_case(text):
    """Capitalise the first letter of each whitespace-separated token.

    The remainder of the token is left untouched, so acronyms and
    camelCase survive. This is intentional: ``"iPhone x"`` becomes
    ``"IPhone X"``.
    """
    if not text:
        return text
    parts = text.split(" ")
    return " ".join(p[0].upper() + p[1:] if p else p for p in parts)


def levenshtein(a, b):
    """Edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current = [i + 1]
        for j, cb in enumerate(b):
            current.append(
                min(
                    previous[j + 1] + 1,
                    current[j] + 1,
                    previous[j] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


__all__ = [
    "parse_duration",
    "truncate",
    "chunk",
    "normalize_whitespace",
    "slugify",
    "title_case",
    "levenshtein",
]