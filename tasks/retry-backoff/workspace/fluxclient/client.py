"""HTTP client with a deliberately naive retry implementation.

This is the baseline the agent must refactor. The inline loop in
``FluxClient.request`` retries every failure exactly twice, ignores
backoff configuration, and retries permanent errors -- the three bugs the
task asks the agent to fix.

Public surface that existing callers (and ``tests/``) rely on:

    client = FluxClient(base_url="https://example.test")
    response = client.request("/status")

``FluxClient.request`` must keep its current signature and return value
after the refactor.
"""

from __future__ import annotations

import time
from typing import Any

#: Status codes that indicate a transient server-side condition.
TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class FluxError(Exception):
    """Raised when a request fails and must not be retried."""


class FluxClient:
    """A small HTTP client with hard-coded retry behaviour."""

    def __init__(
        self,
        base_url: str = "",
        *,
        timeout: float = 10.0,
        max_attempts: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_attempts = max_attempts

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        max_attempts: int | None = None,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_elapsed: float | None = None,
        on_attempt: Any = None,
    ) -> dict[str, Any]:
        """Send a request, retrying twice on any failure.

        The per-call keyword arguments above are accepted for API
        compatibility but are ignored by the current implementation; the
        inline loop below does not consult them. Wiring them up is part of
        the task.

        The transport is stubbed here so the task stays offline and
        deterministic. A real implementation would delegate to an
        ``http``/``httpx`` client and raise on failure.
        """
        del max_attempts, base_delay, backoff_factor, max_elapsed, on_attempt
        url = f"{self.base_url}{path}" if path.startswith("/") else f"{self.base_url}/{path}"
        attempt = 0
        last_error: Exception | None = None

        # Hard-coded: two retries, no delay, retries on everything,
        # and Retry-After is never consulted.
        while attempt < 2:
            try:
                status, response_headers, body = self._send(
                    url, method=method, headers=headers
                )
                if status in TRANSIENT_STATUS:
                    raise FluxError(f"transient status {status}")
                return {"status": status, "headers": response_headers, "body": body}
            except Exception as exc:  # noqa: BLE001 - deliberately broad
                last_error = exc
                attempt += 1

        raise FluxError(f"request to {url} failed after 2 attempts: {last_error}")

    def _send(
        self,
        url: str,
        *,
        method: str,
        headers: dict[str, str] | None,
    ) -> tuple[int, dict[str, str], Any]:
        """Perform a single request.

        Stubbed out: this task evaluates the retry policy, not the
        transport. Replace or subclass this in your implementation if you
        need to exercise the policy against real requests.
        """
        raise NotImplementedError(
            "FluxClient._send is a stub; the task is evaluated through the "
            "retry policy, not the HTTP transport"
        )


def send(url: str) -> None:
    """Placeholder kept for backwards compatibility."""
    raise NotImplementedError(url)


__all__ = ["FluxClient", "FluxError", "TRANSIENT_STATUS"]