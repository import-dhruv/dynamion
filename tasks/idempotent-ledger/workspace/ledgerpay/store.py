"""Append-only payment ledger with known correctness bugs.

Bugs deliberately present in the seed workspace, all of which the task
asks the agent to fix:

* **B1** — every delivery is appended, so a retried webhook double-counts.
* **B2** — an idempotency key reused with a different amount is silently
  accepted rather than rejected.
* **B3** — the entry is appended *before* validation, so a payment that
  fails validation still leaves a ledger entry behind.
* **B4** — there is no synchronisation, so concurrent calls with the same
  key produce multiple entries.

Public names the hidden grader depends on:

    InMemoryLedger, InvalidPayment, ConflictingIdempotencyKey,
    Payment, RecordResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Currencies the ledger accepts.
SUPPORTED_CURRENCIES = frozenset({"USD", "EUR", "GBP", "INR"})


class LedgerError(Exception):
    """Base class for all ledger errors."""


class InvalidPayment(LedgerError):
    """Raised when a payment is malformed or its account is unusable."""


class ConflictingIdempotencyKey(LedgerError):
    """Raised when a key is reused with a different payload."""


class AccountClosed(LedgerError):
    """Raised when a payment targets a closed account."""


@dataclass(frozen=True)
class Payment:
    """An inbound payment from a provider webhook."""

    account: str
    amount: int
    currency: str = "USD"
    idempotency_key: str | None = None

    def fingerprint(self) -> tuple[str, int, str]:
        """Identity of the payment, ignoring the key and any transport id."""
        return (self.account, self.amount, self.currency)


@dataclass(frozen=True)
class RecordResult:
    """Outcome of a ``record_payment`` call."""

    applied: bool
    duplicate: bool
    sequence: int | None = None
    balance: int = 0


@dataclass
class LedgerEntry:
    sequence: int
    account: str
    amount: int
    currency: str
    idempotency_key: str | None = None


@dataclass
class Account:
    balance: int = 0
    closed: bool = False
    currency: str = "USD"
    entries: list[LedgerEntry] = field(default_factory=list)


class InMemoryLedger:
    """In-memory payment ledger.

    Correctness of the accounting invariants is the point of the task;
    persistence is explicitly out of scope.
    """

    def __init__(self) -> None:
        self._accounts: dict[str, Account] = {}
        self._entries: list[LedgerEntry] = []
        self._by_key: dict[str, LedgerEntry] = {}
        self._next_sequence = 1

    # -- public API -------------------------------------------------------

    def open_account(self, account: str, *, currency: str = "USD") -> None:
        if currency not in SUPPORTED_CURRENCIES:
            raise InvalidPayment(f"unsupported currency: {currency!r}")
        self._accounts.setdefault(account, Account(currency=currency))

    def balance(self, account: str) -> int:
        acct = self._accounts.get(account)
        return acct.balance if acct else 0

    def entries(self, account: str | None = None) -> list[LedgerEntry]:
        if account is None:
            return list(self._entries)
        acct = self._accounts.get(account)
        return list(acct.entries) if acct else []

    def record_payment(self, payment: Payment) -> RecordResult:
        """Record a payment, applying it at most once.

        Known-buggy baseline: appends unconditionally and performs no
        synchronisation. See the task prompt for the required behaviour.
        """
        # BUG B4: no lock spans the check and the append.
        if payment.idempotency_key and payment.idempotency_key in self._by_key:
            # BUG B2: a reused key with a different amount should raise,
            # not be accepted as a duplicate.
            return RecordResult(
                applied=False, duplicate=True, balance=self.balance(payment.account)
            )

        # BUG B3: the entry is appended before validation runs.
        entry = LedgerEntry(
            sequence=self._next_sequence,
            account=payment.account,
            amount=payment.amount,
            currency=payment.currency,
            idempotency_key=payment.idempotency_key,
        )
        self._next_sequence += 1
        self._entries.append(entry)

        self._validate(payment)

        acct = self._accounts.setdefault(payment.account, Account())
        acct.balance += payment.amount
        acct.entries.append(entry)
        if payment.idempotency_key:
            self._by_key[payment.idempotency_key] = entry

        return RecordResult(
            applied=True, duplicate=False, sequence=entry.sequence, balance=acct.balance
        )

    # -- internals --------------------------------------------------------

    def _validate(self, payment: Payment) -> None:
        if payment.amount <= 0:
            raise InvalidPayment("amount must be positive")
        if not payment.account:
            raise InvalidPayment("account is required")
        if payment.currency not in SUPPORTED_CURRENCIES:
            raise InvalidPayment(f"unsupported currency: {payment.currency!r}")

        acct = self._accounts.get(payment.account)
        if acct is not None and acct.closed:
            raise AccountClosed(f"account {payment.account} is closed")

    def snapshot(self) -> dict[str, Any]:
        """Debug helper: a comparable view of ledger state."""
        return {
            "entries": [
                (e.sequence, e.account, e.amount, e.currency, e.idempotency_key)
                for e in self._entries
            ],
            "balances": {a: acc.balance for a, acc in self._accounts.items()},
            "next_sequence": self._next_sequence,
        }


__all__ = [
    "InMemoryLedger",
    "LedgerError",
    "InvalidPayment",
    "ConflictingIdempotencyKey",
    "AccountClosed",
    "Payment",
    "RecordResult",
    "LedgerEntry",
    "SUPPORTED_CURRENCIES",
]