"""Invoice pricing engine with money-handling defects.

This module is the one a reviewer must edit **in this file**. Nothing here
is missing: every function exists and works for the common case. The task
is to fix the arithmetic and the ordering, not to add new modules.

The defects are deliberate and are listed in ``KNOWN_DEFECTS`` below. A
successful change fixes all of them without altering any behaviour that
is not listed as a defect.

Callers depend on the parts that already work:

* ``Money`` is a frozen dataclass; ``total`` must keep returning a
  ``Money``, never a float.
* ``format_invoice`` rounds to two decimal places using banker's rounding
  via ``ROUND_HALF_EVEN``, and must continue to do so.
* ``apply_discount`` accepts a percentage as an integer or a fraction, and
  distinguishes them by whether the value exceeds 1. A caller passing
  ``discount=0`` must get no discount; ``discount=100`` must mean 100%.
* ``allocate`` distributes a total across line items so that the parts sum
  exactly to the total. Callers use the remainder, so a penny is never
  created or destroyed.

Do not change any behaviour that is not listed in ``KNOWN_DEFECTS``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Iterable

#: Documented, deliberate defects. Each is a bug the task asks you to fix.
KNOWN_DEFECTS = (
    "apply_discount does not convert whole-number percentages to fractions",
    "allocate drops the rounding remainder instead of distributing it",
    "invoice_total adds credit notes instead of subtracting them",
    "parse_quantity accepts negative quantities",
)


class BillingError(ValueError):
    """Raised for invalid billing input."""


@dataclass(frozen=True)
class Money:
    """An amount in a single currency.

    Frozen: instances are hashable and safe to share.
    """

    amount: Decimal
    currency: str = "USD"

    def __add__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise BillingError(
                f"currency mismatch: {self.currency} + {other.currency}"
            )
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise BillingError(
                f"currency mismatch: {self.currency} - {other.currency}"
            )
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: int | Decimal) -> "Money":
        return Money(self.amount * Decimal(str(factor)), self.currency)

    def scaled(self, factor: Decimal) -> "Money":
        """Scale without rounding."""
        return Money(self.amount * factor, self.currency)

    def rounded(self, places: int = 2) -> "Money":
        quantum = Decimal(1).scaleb(-places)
        return Money(
            self.amount.quantize(quantum, rounding=ROUND_HALF_EVEN), self.currency
        )

    def __repr__(self) -> str:
        return f"Money({self.amount}, {self.currency!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount == other.amount and self.currency == other.currency

    def __hash__(self) -> int:
        return hash((self.amount, self.currency))


@dataclass(frozen=True)
class LineItem:
    description: str
    unit_price: Money
    quantity: int
    #: Fraction already discounted, e.g. 0.10 for 10%.
    discount: Decimal | int = 0


def parse_quantity(raw: str | int) -> int:
    """Parse a quantity from user input.

    Accepts ``"3"``, ``3``, and ``" 3 "``. Raises ``BillingError`` for
    anything that is not a whole number.

    DEFECT: a negative value is accepted rather than rejected.
    """
    if isinstance(raw, bool):
        raise BillingError("quantity must be a number, not a bool")
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    try:
        return int(text)
    except ValueError as exc:
        raise BillingError(f"invalid quantity: {raw!r}") from exc


def apply_discount(amount: Money, discount: Decimal | int) -> Money:
    """Return ``amount`` reduced by ``discount``.

    ``discount`` may be a fraction (``Decimal("0.10")`` or ``0.1``) or a
    whole-number percentage (``10``). Values greater than 1 are read as
    percentages, so ``10`` means 10% and ``0.1`` means 10%. Zero means no
    discount.

    DEFECT: the fraction/percentage distinction is not applied, so
    ``apply_discount(100, 0.1)`` returns 99.9 instead of 90.0.
    """
    # DEFECT: the fraction/percentage distinction is missing, so a
    # fraction like 0.1 is read as 0.1% instead of 10%.
    factor = Decimal(str(discount))
    return amount.scaled(Decimal(1) - factor)


def allocate(total: Money, weights: Iterable[int]) -> list[Money]:
    """Split ``total`` across ``weights`` proportionally.

    The returned amounts sum **exactly** to ``total``. When a weight does
    not divide evenly, the remainder is distributed to the earliest
    entries so no cent is created or destroyed.

    DEFECT: each entry is rounded independently and the remainder is
    discarded, so the parts do not sum to the total.
    """
    weights = list(weights)
    if not weights:
        return []
    if any(w < 0 for w in weights):
        raise BillingError("weights must be non-negative")

    weight_sum = Decimal(sum(weights))
    if weight_sum == 0:
        raise BillingError("weights must not sum to zero")

    return [(total.scaled(Decimal(w) / weight_sum)).rounded() for w in weights]


def line_total(item: LineItem) -> Money:
    """Total for one line item, after its discount."""
    gross = item.unit_price * item.quantity
    if item.discount:
        gross = apply_discount(gross, item.discount)
    return gross


def invoice_total(items: list[LineItem], adjustments: list[Money] | None = None) -> Money:
    """Total for a whole invoice.

    ``adjustments`` are credit notes and refunds, given as the positive
    amount to credit. They must be subtracted from the sum of the line
    items.

    DEFECT: adjustments are added rather than subtracted, so a 10.00 credit
    note on a 100.00 invoice produces 110.00.
    """
    if not items:
        raise BillingError("invoice has no line items")

    currency = items[0].unit_price.currency
    total = Money(Decimal(0), currency)
    for item in items:
        total = total + line_total(item)
    for adjustment in adjustments or []:
        total = total + adjustment
    return total


def format_invoice(items: list[LineItem], currency: str = "USD") -> str:
    """Render an invoice as fixed-width text."""
    lines = [f"{'DESCRIPTION':<32} {'AMOUNT':>12}"]
    lines.append("-" * 45)
    for item in items:
        lines.append(
            f"{item.description[:32]:<32} {str(line_total(item).amount):>12}"
        )
    lines.append("-" * 45)
    lines.append(f"{'TOTAL':<32} {str(invoice_total(items).amount):>12} {currency}")
    return "\n".join(lines)


def to_json(invoice: list[LineItem]) -> str:
    """Serialise line items for the payments API."""
    payload = [
        {
            "description": item.description,
            "unit_price": str(item.unit_price.amount),
            "currency": item.unit_price.currency,
            "quantity": item.quantity,
        }
        for item in invoice
    ]
    return json.dumps(payload, sort_keys=True)


__all__ = [
    "Money",
    "LineItem",
    "BillingError",
    "KNOWN_DEFECTS",
    "parse_quantity",
    "apply_discount",
    "allocate",
    "line_total",
    "invoice_total",
    "format_invoice",
    "to_json",
]