"""Hidden grader tests for the legacy-modernisation task.

Every expected value here was **recorded from the seed implementation**, not
from what the code appears to intend. That is the whole design of the task:
several functions look buggy and are not, and an agent that "fixes" them
produces a diff that looks like an improvement and fails every test.

If you believe one of these expectations is wrong, the recorded output in
``behaviour-golden.json`` next to this file is the authority. Regenerate
it only by running the seed implementation, never by running a
candidate.

Property tests run alongside the recorded cases so that inputs nobody
thought to enumerate are still covered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import legacyutil

GOLDEN = Path(__file__).with_name("behaviour-golden.json")


# ------------------------------------------------------------ public API


def test_public_names_are_preserved():
    """The module's export list is part of its contract."""
    for name in (
        "parse_duration",
        "truncate",
        "chunk",
        "normalize_whitespace",
        "slugify",
        "title_case",
        "levenshtein",
    ):
        assert hasattr(legacyutil, name), f"public name removed: {name}"


def test_golden_file_is_present_and_parses():
    assert GOLDEN.exists(), "recorded behaviour file is missing"
    json.loads(GOLDEN.read_text(encoding="utf-8"))


# ------------------------------------------------------- parse_duration


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1h30m", 5400),
        ("90", 90),
        ("45s", 45),
        ("2d", 172800),
        ("5m", 300),
        (" 5m ", 300),
    ],
)
def test_parse_duration_recognised_inputs(text, expected):
    assert legacyutil.parse_duration(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "10x", None, 123, "   "])
def test_parse_duration_returns_none_rather_than_raising(text):
    """Unparseable input returns None. It must not raise.

    A refactor that introduces a ValueError here breaks every caller that
    uses ``parse_duration(x) or default``.
    """
    assert legacyutil.parse_duration(text) is None


def test_parse_duration_sums_multiple_units():
    assert legacyutil.parse_duration("1h30m10s") == 5400 + 10


# ------------------------------------------------------------- truncate


def test_truncate_shortens_and_appends_ellipsis():
    assert legacyutil.truncate("hello world", 5) == "hell…"


def test_truncate_leaves_short_strings_alone():
    assert legacyutil.truncate("hi", 10) == "hi"
    assert legacyutil.truncate("", 5) == ""


@pytest.mark.parametrize("width", [0, -1, -100, None])
def test_truncate_never_raises_on_non_positive_width(width):
    """A non-positive width returns the input unchanged.

    Counter-intuitive, and callers depend on it. Slicing to a negative
    width would silently return the wrong string.
    """
    assert legacyutil.truncate("hello", width) == "hello"


# ----------------------------------------------------------------- chunk


def test_chunk_keeps_the_trailing_partial_chunk():
    assert legacyutil.chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


@pytest.mark.parametrize("size", [0, -1])
def test_chunk_with_non_positive_size_returns_one_chunk(size):
    """Callers use a non-positive size to mean 'do not chunk'."""
    assert legacyutil.chunk([1, 2, 3], size) == [[1, 2, 3]]


def test_chunk_of_empty_is_empty():
    assert legacyutil.chunk([], 3) == []


def test_chunk_exact_multiple_has_no_trailing_empty_chunk():
    assert legacyutil.chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


# ----------------------------------------------------- normalize_whitespace


def test_normalize_whitespace_preserves_leading_and_trailing_space():
    """The seed keeps exactly one leading and one trailing space."""
    assert legacyutil.normalize_whitespace("  a   b  ") == " a b "


def test_normalize_whitespace_collapses_internal_runs():
    assert legacyutil.normalize_whitespace("a \t\n b") == "a b"


def test_normalize_whitespace_passes_through_none_and_empty():
    assert legacyutil.normalize_whitespace(None) is None
    assert legacyutil.normalize_whitespace("") == ""


# --------------------------------------------------------------- slugify


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello World!", "helloworld"),
        ("iPhone 15 Pro", "iphone15pro"),
        ("a", "a"),
        ("", ""),
        ("!!!", ""),
    ],
)
def test_slugify_recorded_behaviour(text, expected):
    """A string with no alphanumerics yields "", not a placeholder."""
    assert legacyutil.slugify(text) == expected


def test_slugify_is_idempotent():
    for text in ["Hello World", "a b", "!!!", ""]:
        once = legacyutil.slugify(text)
        assert legacyutil.slugify(text) == once


# ------------------------------------------------------------ title_case


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello world", "Hello World"),
        # Recorded, not intended: the rest of the token is untouched.
        ("iPhone x", "IPhone X"),
        ("MIXED case HERE", "MIXED Case HERE"),
        ("a", "A"),
        ("", ""),
    ],
)
def test_title_case_preserves_token_interiors(text, expected):
    """title_case must NOT lowercase the tail of a token."""
    assert legacyutil.title_case(text) == expected


# ------------------------------------------------------------ levenshtein


def test_levenshtein_basics():
    assert legacyutil.levenshtein("kitten", "sitting") == 3
    assert legacyutil.levenshtein("same", "same") == 0
    assert legacyutil.levenshtein("", "abc") == 3


# ---------------------------------------------------------- properties


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(st.text())
def test_slugify_never_raises_and_is_alphanumeric(text):
    result = legacyutil.slugify(text)
    assert result == result.lower()
    assert all(c.isalnum() for c in result)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(st.text(), st.integers(min_value=-5, max_value=50))
def test_truncate_never_raises(text, width):
    """Property form of the non-positive-width contract."""
    result = legacyutil.truncate(text, width)
    assert isinstance(result, str)
    if width <= 0:
        assert result == text


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.integers(), max_size=40), st.integers(min_value=0, max_value=8))
def test_chunk_preserves_all_elements(items, size):
    chunks = legacyutil.chunk(items, size)
    flattened = [x for c in chunks for x in c]
    assert flattened == list(items), "chunking must not drop or reorder elements"
    if size >= 1:
        assert all(len(c) <= size for c in chunks)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(st.text())
def test_normalize_whitespace_is_idempotent(text):
    once = legacyutil.normalize_whitespace(text)
    assert legacyutil.normalize_whitespace(once) == once


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(st.text())
def test_parse_duration_never_raises(text):
    result = legacyutil.parse_duration(text)
    assert result is None or isinstance(result, int)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(st.text())
def test_title_case_never_lowercases_token_interiors(text):
    result = legacyutil.title_case(text)
    assert isinstance(result, str)
    if result:
        assert result[0] == result[0].upper() or not result[0].isalpha()