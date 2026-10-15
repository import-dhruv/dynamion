"""Hidden grader tests for the idempotent-ledger task.

Not present in the agent's workspace; copied in only after the agent's
turn ends. Covers the four seeded bugs plus the invariants that a partial
fix typically breaks:

  * B1 double-counting on retry
  * B2 conflicting idempotency-key reuse
  * B3 entries surviving a failed validation
  * B4 duplicate entries under concurrency
  * sequence monotonicity across rejected duplicates
  * closed-account rejection
"""

from __future__ import annotations

import threading

import pytest

from ledgerpay import (
    AccountClosed,
    ConflictingIdempotencyKey,
    InMemoryLedger,
    InvalidPayment,
    Payment,
)


@pytest.fixture
def ledger():
    store = InMemoryLedger()
    store.open_account("acct_1")
    return store


# ------------------------------------------------------------ idempotency


def test_public_api_surface_is_intact():
    """Every name the package re-exports must still exist.

    Added after a real run rewrote ``store.py`` from scratch and kept only
    three of the eight re-exported names. The import in ``__init__.py``
    then failed, and because the failure surfaced as a pytest collection
    error the whole run was unscoreable.

    A prompt that asks the agent to preserve an API is not a safeguard.
    This test is.
    """
    import ledgerpay

    expected = {
        "InMemoryLedger",
        "InvalidPayment",
        "ConflictingIdempotencyKey",
        "AccountClosed",
        "LedgerError",
        "Payment",
        "RecordResult",
        "LedgerEntry",
    }
    missing = {name for name in expected if not hasattr(ledgerpay, name)}
    assert not missing, f"public API names removed: {sorted(missing)}"


def test_payment_remains_constructible_with_keyword_arguments():
    """`Payment` is a frozen dataclass; keep its field names stable."""
    payment = Payment(account="acct_1", amount=100, currency="USD", idempotency_key="k")
    assert payment.amount == 100
    assert payment.fingerprint() == ("acct_1", 100, "USD")


def test_single_payment_applies(ledger):
    result = ledger.record_payment(Payment("acct_1", 100, idempotency_key="k1"))
    assert result.applied is True
    assert result.duplicate is False
    assert ledger.balance("acct_1") == 100


def test_retry_with_same_key_is_duplicate(ledger):
    ledger.record_payment(Payment("acct_1", 100, idempotency_key="k1"))
    result = ledger.record_payment(Payment("acct_1", 100, idempotency_key="k1"))
    assert result.duplicate is True
    assert result.applied is False
    assert ledger.balance("acct_1") == 100, "retry must not double-count"


def test_retry_without_key_identical_payload_is_duplicate(ledger):
    ledger.record_payment(Payment("acct_1", 100))
    result = ledger.record_payment(Payment("acct_1", 100))
    assert result.duplicate is True
    assert ledger.balance("acct_1") == 100


def test_distinct_payments_both_apply(ledger):
    ledger.record_payment(Payment("acct_1", 100, idempotency_key="k1"))
    ledger.record_payment(Payment("acct_1", 50, idempotency_key="k2"))
    assert ledger.balance("acct_1") == 150


def test_key_reuse_with_different_amount_raises(ledger):
    ledger.record_payment(Payment("acct_1", 100, idempotency_key="k1"))
    before = ledger.balance("acct_1")
    with pytest.raises(ConflictingIdempotencyKey):
        ledger.record_payment(Payment("acct_1", 999, idempotency_key="k1"))
    assert ledger.balance("acct_1") == before, "conflict must not mutate the ledger"


def test_key_reuse_with_different_account_raises(ledger):
    ledger.open_account("acct_2")
    ledger.record_payment(Payment("acct_1", 100, idempotency_key="k1"))
    with pytest.raises(ConflictingIdempotencyKey):
        ledger.record_payment(Payment("acct_2", 100, idempotency_key="k1"))


# ------------------------------------------------------------- validation


def test_non_positive_amount_rejected_without_side_effects(ledger):
    before = len(ledger.entries())
    with pytest.raises(InvalidPayment):
        ledger.record_payment(Payment("acct_1", 0, idempotency_key="bad"))
    with pytest.raises(InvalidPayment):
        ledger.record_payment(Payment("acct_1", -5, idempotency_key="bad"))
    assert len(ledger.entries()) == before, "invalid payment left a ledger entry"
    assert ledger.balance("acct_1") == 0


def test_unknown_currency_rejected_without_side_effects(ledger):
    before = len(ledger.entries())
    with pytest.raises(InvalidPayment):
        ledger.record_payment(Payment("acct_1", 100, "XYZ", idempotency_key="bad"))
    assert len(ledger.entries()) == before


def test_closed_account_refuses_payment(ledger):
    acct = ledger._accounts["acct_1"]
    acct.closed = True
    before = len(ledger.entries())
    with pytest.raises(AccountClosed):
        ledger.record_payment(Payment("acct_1", 100, idempotency_key="k9"))
    assert len(ledger.entries()) == before, "closed-account payment left an entry"


# ------------------------------------------------------------- sequencing


def test_sequence_numbers_are_monotonic(ledger):
    seqs = []
    for i in range(5):
        result = ledger.record_payment(Payment("acct_1", 10, idempotency_key=f"k{i}"))
        seqs.append(result.sequence)
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == 5


def test_duplicate_consumes_no_sequence(ledger):
    first = ledger.record_payment(Payment("acct_1", 100, idempotency_key="k1"))
    ledger.record_payment(Payment("acct_1", 100, idempotency_key="k1"))
    second = ledger.record_payment(Payment("acct_1", 50, idempotency_key="k2"))
    assert second.sequence == first.sequence + 1, "duplicate consumed a sequence number"


# ------------------------------------------------------------ concurrency


def test_concurrent_same_key_creates_one_entry(ledger):
    """Race detection is inherently probabilistic.

    A thread barrier aligns the workers at the start of ``record_payment``
    and repeated rounds widen the check-then-append window, so an
    unsynchronised implementation is caught with high probability across
    enough rounds. A correctly locked implementation passes every round
    because the lock serialises the whole critical section regardless of
    scheduling.
    """
    THREADS = 8
    ROUNDS = 25
    failures: list[str] = []

    for round_index in range(ROUNDS):
        store = InMemoryLedger()
        store.open_account("acct_1")
        barrier = threading.Barrier(THREADS)
        errors: list[BaseException] = []

        def worker(store=store, barrier=barrier):
            try:
                # Force every thread to arrive before any proceeds.
                barrier.wait(timeout=10)
                store.record_payment(Payment("acct_1", 100, idempotency_key="race"))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            failures.append(f"round {round_index}: raised {errors[:2]}")
            continue
        if store.balance("acct_1") != 100:
            failures.append(
                f"round {round_index}: balance {store.balance('acct_1')} != 100"
            )
        if len(store.entries("acct_1")) != 1:
            failures.append(
                f"round {round_index}: {len(store.entries('acct_1'))} entries, expected 1"
            )

    assert not failures, "race detected in unsynchronised implementation: " + "; ".join(
        failures[:5]
    )


def test_store_synchronises_check_and_append(ledger):
    """Deterministic check that the implementation *has* synchronisation.

    Why this exists: the race in the test above cannot be reliably
    triggered for a fast in-memory implementation under CPython. The
    check-then-append window is measured in microseconds while the GIL
    switches on a 5ms interval, so an unsynchronised implementation
    usually passes by luck. A test that only sometimes catches the bug
    gives false confidence, so this asserts the structural property
    directly instead.

    This is white-box: it inspects the store for a lock rather than
    observing behaviour. That is a deliberate trade. Black-box race
    detection is unreliable here, and a benchmark that awards credit for
    an undetected bug is worse than one that inspects internals.
    """
    store = InMemoryLedger()
    has_lock = any(
        isinstance(getattr(store, attr, None), type(threading.Lock()))
        or hasattr(getattr(store, attr, None), "acquire")
        for attr in dir(store)
        if not attr.startswith("__")
    )
    assert has_lock, (
        "InMemoryLedger exposes no lock; an unsynchronised check-then-append "
        "sequence can create duplicate entries under concurrency even when "
        "the race test above happens not to trigger"
    )


def test_concurrent_distinct_keys_all_apply(ledger):
    def worker(i: int):
        ledger.record_payment(Payment("acct_1", 10, idempotency_key=f"c{i}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert ledger.balance("acct_1") == 100
    assert len({e.sequence for e in ledger.entries("acct_1")}) == 10, "sequence collision"