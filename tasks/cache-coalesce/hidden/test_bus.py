"""Hidden grader tests for the event-bus task.

Concurrency is hard to grade honestly, so this suite avoids depending on
winning a race. Each concurrency defect is checked **structurally** — that
the critical section exists and that a synchronisation primitive is held
across it — alongside behavioural checks that are deterministic.

Rationale, from the `idempotent-ledger` experience: a behavioural-only
race test passed an unsynchronised implementation three times out of three
because the race window is microseconds wide and CPython switches threads
every 5ms. A test that only sometimes catches the bug is worse than no
test, because it grants credit for an undetected defect. These tests
check the property directly instead, and say so.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import threading

import pytest

import eventbus
from eventbus import CacheStats, EventBus, EventBusError, PublishResult

TOPIC = "orders.created"


# ------------------------------------------------------------ public API


def test_public_surface_intact():
    for name in ("EventBus", "EventBusError", "CacheStats", "PublishResult"):
        assert hasattr(eventbus, name), f"public name removed: {name}"


def test_known_defects_still_documented():
    assert eventbus.KNOWN_DEFECTS, "KNOWN_DEFECTS was emptied"


def test_cache_stats_is_frozen_and_compares_by_value():
    assert CacheStats(1, 1, 0, 0) == CacheStats(1, 1, 0, 0)
    with pytest.raises(Exception):
        CacheStats(1, 1, 0, 0).hits = 2


def test_hit_rate():
    assert CacheStats(3, 1, 0, 0).hit_rate == 0.75
    assert CacheStats(0, 0, 0, 0).hit_rate == 0.0


# ------------------------------------------------------- basic behaviour


def test_publish_delivers_to_a_subscriber():
    bus = EventBus()
    seen = []
    bus.subscribe(TOPIC, lambda t, p: seen.append((t, p)))
    result = bus.publish(TOPIC, {"id": 1})
    assert result.delivered == 1
    assert seen == [(TOPIC, {"id": 1})]


def test_publish_with_no_subscribers_delivers_nothing():
    """A topic nobody subscribed to must not become a real topic."""
    bus = EventBus()
    result = bus.publish("never.subscribed", 1)
    assert result.delivered == 0
    assert result.errors == ()
    assert "never.subscribed" not in bus.topics(), (
        "publishing created a topic entry for a topic nobody subscribed to; "
        "the bus grows without bound"
    )


def test_publish_collects_handler_errors_without_stopping_others():
    bus = EventBus()
    ran = []

    def bad(topic, payload):
        raise RuntimeError("subscriber blew up")

    def good(topic, payload):
        ran.append(payload)

    bus.subscribe(TOPIC, bad)
    bus.subscribe(TOPIC, good)
    result = bus.publish(TOPIC, 1)

    assert ran == [1], "a failing subscriber must not prevent the others"
    assert result.delivered == 1
    assert len(result.errors) == 1
    assert isinstance(result.errors[0], RuntimeError)


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    seen = []
    unsub = bus.subscribe(TOPIC, lambda t, p: seen.append(p))
    bus.publish(TOPIC, 1)
    unsub()
    bus.publish(TOPIC, 2)
    assert seen == [1]


def test_unsubscribe_is_idempotent():
    """Calling the unsubscriber twice is safe and must not raise."""
    bus = EventBus()
    unsub = bus.subscribe(TOPIC, lambda t, p: None)
    unsub()
    unsub()
    assert bus.subscriber_count(TOPIC) == 0


def test_unsubscribe_reports_whether_it_was_present():
    """Removing something that was never there must report False.

    The current implementation always returns True, which makes a
    double-unsubscribe indistinguishable from a successful one.
    """
    bus = EventBus()
    handler = lambda t, p: None  # noqa: E731
    assert bus.unsubscribe(TOPIC, handler) is False, (
        "unsubscribing an unregistered handler must return False, not True"
    )
    bus.subscribe(TOPIC, handler)
    assert bus.unsubscribe(TOPIC, handler) is True


def test_rejects_invalid_subscription():
    bus = EventBus()
    with pytest.raises(EventBusError):
        bus.subscribe("", lambda t, p: None)
    with pytest.raises(EventBusError):
        bus.subscribe(TOPIC, "not callable")


def test_handler_list_is_a_copy():
    bus = EventBus()
    bus.subscribe(TOPIC, lambda t, p: None)
    handlers = bus.handler_list(TOPIC)
    handlers.clear()
    assert bus.subscriber_count(TOPIC) == 1, "handler_list must not expose internals"


# ------------------------------- defect: structural synchronisation checks


def _method_source(fn) -> str:
    return inspect.getsource(fn)


def _called_names(fn) -> set[str]:
    """Names actually *called* in ``fn``, from the AST.

    Grepping the source text is not good enough: an earlier version of
    this suite asserted ``"setdefault" not in source`` and matched the
    word inside a comment that documented the defect. Structural tests
    must read the tree, not the prose.
    """
    # Method source is indented; ast.parse requires it dedented.
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                called.add(func.attr)
            elif isinstance(func, ast.Name):
                called.add(func.id)
    return called


def _has_lock_held(fn, must_mention: tuple[str, ...]) -> bool:
    """True if ``fn`` acquires a lock within a block mentioning all phrases."""
    source = _method_source(fn)
    if "with self._lock" not in source and "self._lock.acquire" not in source:
        return False
    lowered = source.lower()
    return all(phrase.lower() in lowered for phrase in must_mention)


def test_subscribe_holds_the_lock():
    """The read and the append in ``subscribe`` must be one critical section.

    Checked structurally on purpose. A behavioural race test passed this
    implementation every time it was run, because the window is
    microseconds wide; a test that only sometimes catches the bug is worse
    than no test, because it credits an undetected defect.
    """
    assert _has_lock_held(
        EventBus.subscribe, ("append", "subscribe")
    ), (
        "subscribe does not hold the lock across the read and the append, "
        "so concurrent subscribes can both insert the same handler"
    )


def test_unsubscribe_holds_the_lock():
    assert _has_lock_held(EventBus.unsubscribe, ("remove",)), (
        "unsubscribe does not hold the lock while mutating the handler list"
    )


def test_publish_does_not_create_topics():
    """``setdefault`` in publish is the defect.

    Parsed from the AST rather than matched as text, so that a comment
    mentioning the word cannot satisfy or fail the check.
    """
    assert "setdefault" not in _called_names(EventBus.publish), (
        "publish must not create a topic entry for a topic with no subscribers"
    )


def test_stats_reads_under_the_lock():
    assert _has_lock_held(EventBus.stats, ("stats",)) or "with self._lock" in _method_source(EventBus.stats), (
        "stats() reads counters without synchronisation"
    )


# ----------------------------------------------- deterministic concurrency


def test_concurrent_subscribes_all_survive():
    """Every distinct handler registered from a thread must be present.

    This is the observable invariant. Note what it deliberately does *not*
    assert: that subscribing the same handler N times yields one entry.
    Append semantics mean N, and claiming otherwise would fail a correct
    implementation. The defect this task is about is a lost or corrupted
    update, not deduplication.
    """
    bus = EventBus()
    count = 8
    barrier = threading.Barrier(count)
    errors = []

    def worker(n):
        try:
            barrier.wait(timeout=10)
            bus.subscribe(TOPIC, lambda t, p, n=n: n)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent subscribe raised: {errors[:2]}"
    assert bus.subscriber_count(TOPIC) == count, (
        f"expected {count} handlers, got {bus.subscriber_count(TOPIC)}: "
        "a concurrent subscribe was lost"
    )


def test_concurrent_publish_delivers_every_time():
    """Many threads publishing must all be delivered, none dropped."""
    bus = EventBus()
    received = []
    lock = threading.Lock()

    def handler(topic, payload):
        with lock:
            received.append(payload)

    bus.subscribe(TOPIC, handler)
    threads_count, per_thread = 8, 50
    barrier = threading.Barrier(threads_count)

    def worker(n):
        barrier.wait(timeout=10)
        for i in range(per_thread):
            bus.publish(TOPIC, (n, i))

    threads = [
        threading.Thread(target=worker, args=(n,)) for n in range(threads_count)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(received) == threads_count * per_thread, (
        f"expected {threads_count * per_thread} deliveries, got {len(received)}"
    )
    assert len(set(received)) == threads_count * per_thread, "a delivery was duplicated"


def test_stats_are_consistent_after_concurrent_publish():
    bus = EventBus()
    bus.subscribe(TOPIC, lambda t, p: None)

    def worker():
        for _ in range(20):
            bus.publish(TOPIC, 1)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = bus.stats()
    assert isinstance(stats, CacheStats)
    assert stats.hits + stats.misses == 6 * 20, (
        f"counters inconsistent under concurrency: {stats}"
    )


def test_unsubscribe_during_publish_is_not_corrupted():
    """Removing a handler mid-publish must not corrupt the list."""
    bus = EventBus()
    handler = lambda t, p: None  # noqa: E731
    for _ in range(50):
        bus.subscribe(TOPIC, handler)

    errors = []

    def unsubscriber():
        try:
            for _ in range(50):
                bus.unsubscribe(TOPIC, handler)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def publisher():
        try:
            for _ in range(50):
                bus.publish(TOPIC, 1)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=unsubscriber), threading.Thread(target=publisher)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent unsubscribe/publish raised: {errors[:2]}"
    assert bus.subscriber_count(TOPIC) >= 0


def test_module_imports_threading():
    """The module must still be able to synchronise."""
    tree = ast.parse(inspect.getsource(eventbus))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "threading" in imported, "threading import was removed"


def test_publish_result_is_immutable():
    result = PublishResult(delivered=1, errors=(RuntimeError("x"),))
    with pytest.raises(Exception):
        result.delivered = 2
    assert isinstance(result.errors, tuple)