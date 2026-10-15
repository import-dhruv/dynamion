"""Event bus with a subscriber cache that has four concurrency defects.

Edit this file. The bugs are all about what happens when two threads do
the same thing at once, which is exactly the class of defect that unit
tests written by the author will never catch, because the author was not
thinking about concurrency when they wrote them.

The four defects are listed in ``KNOWN_DEFECTS``.

Callers depend on the parts that already work:

* ``subscribe`` returns an unsubscribe callable. Calling it twice is safe
  and must be idempotent.
* ``publish`` returns the number of handlers that ran, and must return 0
  when there are none.
* Handler exceptions are collected rather than raised: one bad subscriber
  must not prevent the others from running. The exceptions are available
  on the returned ``PublishResult``.
* ``CacheStats`` is a frozen dataclass and is compared by value in tests.

The event bus is single-process and in-memory. There is no I/O and no
network.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

#: Documented, deliberate defects.
KNOWN_DEFECTS = (
    "subscribe mutates the subscriber list without holding the lock",
    "unsubscribe does not check whether it already removed the handler",
    "publish snapshots the handler list into a defaultdict, creating entries",
    "stats() reads counters without synchronisation",
)


class EventBusError(RuntimeError):
    """Raised for invalid bus usage."""


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    subscribes: int
    unsubscribes: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass(frozen=True)
class PublishResult:
    """What happened during one ``publish``."""

    delivered: int
    errors: tuple[BaseException, ...] = ()


Handler = Callable[[str, Any], None]


class EventBus:
    """A synchronous in-process publish/subscribe bus."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # topic -> list of handlers
        self._subs: dict[str, list[Handler]] = {}
        self._hits = 0
        self._misses = 0
        self._subscribes = 0
        self._unsubscribes = 0

    # -- subscription -----------------------------------------------------

    def subscribe(self, topic: str, handler: Handler) -> Callable[[], None]:
        """Register ``handler`` for ``topic``; returns an unsubscriber."""
        if not topic:
            raise EventBusError("topic must be non-empty")
        if not callable(handler):
            raise EventBusError("handler must be callable")

        # DEFECT: the read-then-append is not inside the lock, so two
        # threads subscribing the same handler can both observe "not
        # present" and both append, giving duplicate deliveries.
        self._subscribes += 1
        self._subs.setdefault(topic, []).append(handler)

        def unsubscribe() -> None:
            self.unsubscribe(topic, handler)

        return unsubscribe

    def unsubscribe(self, topic: str, handler: Handler) -> bool:
        """Remove ``handler`` from ``topic``. True if it was present.

        DEFECT: no membership check and no lock, so unsubscribing a handler
        that is not registered returns True, and concurrent unsubscribes can
        leave the list corrupted.
        """
        self._unsubscribes += 1
        handlers = self._subs.get(topic)
        if handlers is None:
            return True
        try:
            handlers.remove(handler)
        except ValueError:
            return True
        return True

    def subscriber_count(self, topic: str) -> int:
        with self._lock:
            return len(self._subs.get(topic, ()))

    # -- publishing -------------------------------------------------------

    def publish(self, topic: str, payload: Any) -> PublishResult:
        """Deliver ``payload`` to every handler for ``topic``.

        Handler exceptions are collected, not raised: a failing subscriber
        must not stop the others.
        """
        with self._lock:
            # DEFECT: setdefault creates an empty list for every topic ever
            # published to, so the bus grows without bound and a topic
            # nobody subscribed to looks like a real topic.
            handlers = self._subs.setdefault(topic, [])
            snapshot = list(handlers)

        if not snapshot:
            self._misses += 1
            return PublishResult(delivered=0, errors=())

        self._hits += 1
        delivered = 0
        errors: list[BaseException] = []
        for handler in snapshot:
            try:
                handler(topic, payload)
                delivered += 1
            except Exception as exc:  # noqa: BLE001 - collected by contract
                errors.append(exc)
        return PublishResult(delivered=delivered, errors=tuple(errors))

    # -- introspection ----------------------------------------------------

    def topics(self) -> set[str]:
        with self._lock:
            return set(self._subs)

    def stats(self) -> CacheStats:
        """Snapshot the counters.

        DEFECT: read without synchronisation, so a concurrent publish can
        be observed as a torn or stale value.
        """
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            subscribes=self._subscribes,
            unsubscribes=self._unsubscribes,
        )

    def handler_list(self, topic: str) -> list[Handler]:
        """Return a copy of the handler list, for inspection."""
        with self._lock:
            return list(self._subs.get(topic, ()))


__all__ = [
    "EventBus",
    "EventBusError",
    "CacheStats",
    "PublishResult",
    "Handler",
    "KNOWN_DEFECTS",
]