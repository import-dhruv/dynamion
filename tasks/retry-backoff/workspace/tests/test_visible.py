"""Visible tests for the retry-policy task.

These are deliberately weaker than the hidden grader tests. Passing every
test here is necessary but not sufficient: the hidden suite additionally
covers the max_elapsed deadline, callback invocation on the final attempt,
and transport independence.

Do not modify this file. The grader runs its own copy.
"""

from __future__ import annotations

import pytest

from fluxclient.client import FluxError, FluxClient


class StubClient(FluxClient):
    """A client whose transport replays a scripted sequence of outcomes.

    Each entry in ``script`` is either an ``(status, headers)`` tuple for a
    successful response or an exception instance to raise.
    """

    def __init__(self, script, **kwargs):
        super().__init__(**kwargs)
        self.script = list(script)
        self.calls = 0

    def _send(self, url, *, method, headers):
        self.calls += 1
        outcome = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        status, response_headers = outcome
        return status, response_headers, b"body"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Replace time.sleep so tests do not actually wait."""
    slept: list[float] = []

    def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr("fluxclient.client.time.sleep", fake_sleep, raising=False)
    monkeypatch.setattr("fluxclient.retry.time.sleep", fake_sleep, raising=False)
    monkeypatch.setattr("time.sleep", fake_sleep)
    return slept


def test_transient_status_is_retried():
    client = StubClient([(503, {}), (200, {})])
    response = client.request("/x", max_attempts=3)
    assert response["status"] == 200
    assert client.calls == 2


def test_permanent_status_is_raised_immediately():
    client = StubClient([(404, {})])
    with pytest.raises(FluxError):
        client.request("/x", max_attempts=5)
    assert client.calls == 1, "4xx must not be retried"


def test_max_attempts_one_disables_retry():
    client = StubClient([(503, {})])
    with pytest.raises(FluxError):
        client.request("/x", max_attempts=1)
    assert client.calls == 1