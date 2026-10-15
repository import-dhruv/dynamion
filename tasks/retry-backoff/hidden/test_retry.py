"""Hidden grader tests for the retry-policy task.

This file is never present in the agent's workspace. It is copied in only
after the agent's turn ends, which is what makes the score meaningful: an
agent cannot satisfy these tests without implementing the behaviour.

Coverage beyond the visible tests:
  * exponential backoff arithmetic
  * Retry-After precedence, including malformed values
  * the max_elapsed deadline
  * callback invocation (and non-invocation on the final attempt)
  * ValueError on invalid configuration
  * transport independence (the policy must work with no HTTP client)
"""

from __future__ import annotations

import inspect

import pytest

from fluxclient.client import FluxClient, FluxError



class StubClient(FluxClient):
    def __init__(self, script, **kwargs):
        super().__init__(**kwargs)
        self.script = list(script)
        self.calls = 0
        self.attempts_seen = []

    def _send(self, url, *, method, headers):
        self.calls += 1
        self.attempts_seen.append(self.calls)
        outcome = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        status, response_headers = outcome
        return status, response_headers, b"body"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    slept: list[float] = []

    def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr("time.sleep", fake_sleep)
    return slept


# ---------------------------------------------------------------- retries


def test_transient_codes_retried():
    for code in (408, 429, 500, 502, 503, 504):
        client = StubClient([(code, {}), (200, {})])
        assert client.request("/x", max_attempts=3)["status"] == 200
        assert client.calls == 2, f"{code} should be retried"


def test_permanent_codes_not_retried():
    for code in (400, 401, 403, 404, 409, 422):
        client = StubClient([(code, {})])
        with pytest.raises(FluxError):
            client.request("/x", max_attempts=5)
        assert client.calls == 1, f"{code} must not be retried"


def test_transport_errors_retried():
    client = StubClient([ConnectionResetError("reset"), (200, {})])
    assert client.request("/x", max_attempts=3)["status"] == 200
    assert client.calls == 2


def test_exhausted_attempts_raises():
    client = StubClient([(503, {})])
    with pytest.raises(FluxError):
        client.request("/x", max_attempts=2)
    assert client.calls == 2


# ----------------------------------------------------------------- config


def test_max_attempts_below_one_raises_value_error():
    client = StubClient([(200, {})])
    with pytest.raises(ValueError):
        client.request("/x", max_attempts=0)


def test_max_attempts_one_disables_retry():
    client = StubClient([(503, {})])
    with pytest.raises(FluxError):
        client.request("/x", max_attempts=1)
    assert client.calls == 1


# ---------------------------------------------------------------- backoff


def test_exponential_backoff_delays(_no_sleep):
    client = StubClient([(503, {})])
    with pytest.raises(FluxError):
        client.request("/x", max_attempts=4, base_delay=1.0, backoff_factor=2.0)
    assert _no_sleep == [1.0, 2.0, 4.0], f"unexpected backoff: {_no_sleep}"


def test_first_retry_uses_base_delay(_no_sleep):
    client = StubClient([(503, {})])
    with pytest.raises(FluxError):
        client.request("/x", max_attempts=2, base_delay=0.25, backoff_factor=3.0)
    assert _no_sleep == [0.25]


# ------------------------------------------------------------ retry-after


def test_retry_after_takes_precedence(_no_sleep):
    client = StubClient([(429, {"Retry-After": "7"}), (200, {})])
    assert client.request("/x", max_attempts=2, base_delay=1.0)["status"] == 200
    assert _no_sleep == [7.0], f"Retry-After should win, got {_no_sleep}"


@pytest.mark.parametrize("bad", ["not-a-number", "-5", "", "1.2.3"])
def test_malformed_retry_after_falls_back(_no_sleep, bad):
    # Header name casing must not matter; the value must fall back to
    # computed backoff rather than raising.
    client = StubClient([(503, {"Retry-After": bad}), (200, {})])
    assert client.request("/x", max_attempts=2, base_delay=2.0)["status"] == 200
    assert _no_sleep == [2.0], f"malformed Retry-After must fall back, got {_no_sleep}"


# -------------------------------------------------------------- max_elapsed


def test_max_elapsed_gives_up_without_sleeping_past_deadline(_no_sleep):
    client = StubClient([(503, {})])
    with pytest.raises(FluxError):
        client.request(
            "/x", max_attempts=10, base_delay=10.0, backoff_factor=2.0, max_elapsed=15.0
        )
    # Only the first sleep fits inside the 15s deadline.
    assert _no_sleep == [10.0], f"expected one sleep before deadline, got {_no_sleep}"
    assert client.calls == 2, "must not attempt again after the deadline passes"


# ---------------------------------------------------------------- callback


def test_callback_receives_attempts_and_delays(_no_sleep):
    client = StubClient([(503, {})])
    seen = []
    with pytest.raises(FluxError):
        client.request(
            "/x",
            max_attempts=3,
            base_delay=1.0,
            backoff_factor=2.0,
            on_attempt=lambda attempt, exc, delay: seen.append((attempt, delay)),
        )
    assert len(seen) == 2, "callback must fire once per retry, not per attempt"
    assert [a for a, _ in seen] == [1, 2]
    assert [d for _, d in seen] == [1.0, 2.0]


def test_callback_not_invoked_on_final_attempt(_no_sleep):
    client = StubClient([(200, {})])
    seen = []
    client.request(
        "/x", max_attempts=3, on_attempt=lambda a, e, d: seen.append(a)
    )
    assert seen == [], "no callback when the first attempt succeeds"


# ------------------------------------------------------- transport independence


def test_policy_usable_without_http_client():
    """The policy must be separable from the HTTP transport.

    Deliberately does *not* guess a module name. An earlier version of
    this test looked for ``fluxclient.retry`` and friends, which scored
    zero for an agent that wrote a perfectly good transport-independent
    policy under a different filename. Guessing names tests transcription,
    not capability, so this searches the package for any policy-like module
    that does not depend on the client.
    """
    import ast
    import pathlib
    import re

    workspace = pathlib.Path(inspect.getfile(FluxClient)).parent.parent
    candidates = []

    for path in sorted(workspace.rglob("*.py")):
        rel = path.relative_to(workspace).as_posix()
        # Test files are the grader's own; they are not candidates.
        if rel.startswith("tests/") or rel.startswith("hidden/"):
            continue
        if path.name == "client.py":
            continue

        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        imports_client = any(
            (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").endswith("client")
            )
            or (
                isinstance(node, ast.Import)
                and any(a.name.endswith("client") for a in node.names)
            )
            for node in ast.walk(tree)
        )
        if imports_client:
            continue

        # A policy module is expected to talk about retry behaviour.
        if re.search(r"\b(retry|attempt|backoff)\b", source, re.I):
            candidates.append(rel)

    assert candidates, (
        "no transport-independent retry policy module found in the workspace; "
        "the retry logic appears to be entangled with FluxClient. Modules present: "
        + ", ".join(
            sorted(
                p.relative_to(workspace).as_posix()
                for p in workspace.rglob("*.py")
                if not p.relative_to(workspace).as_posix().startswith(("tests/", "hidden/"))
            )
        )
    )