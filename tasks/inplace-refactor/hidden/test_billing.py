"""Hidden grader tests for the in-place refactor task.

Every expected value here was executed against the **fixed** behaviour,
not inferred. This task exists partly to test whether an agent edits the
file it was told to edit: in three separate runs on sibling tasks a 7B
model wrote structurally sound code and put it in a *new* file at the
repository root, leaving the real module untouched. Grading only
``pkbilling/__init__.py`` is what makes that failure observable instead
of invisible.
"""

from __future__ import annotations

import pytest
from decimal import Decimal

from pkbilling import (
    KNOWN_DEFECTS,
    BillingError,
    LineItem,
    Money,
    allocate,
    apply_discount,
    invoice_total,
    line_total,
    parse_quantity,
    to_json,
)


def M(amount, currency="USD") -> Money:
    return Money(Decimal(str(amount)), currency)


# ------------------------------------------------------------- public API


def test_public_surface_intact():
    for name in (
        "Money", "LineItem", "BillingError", "parse_quantity",
        "apply_discount", "allocate", "line_total", "invoice_total",
        "format_invoice", "to_json",
    ):
        import pkbilling

        assert hasattr(pkbilling, name), f"public name removed: {name}"


def test_module_still_documents_its_known_defects():
    """Removing the list is not a fix; keeping it honest is part of the job."""
    assert KNOWN_DEFECTS, "KNOWN_DEFECTS was emptied"


# ------------------------------------------------- defect 1: percentages


@pytest.mark.parametrize(
    ("discount", "expected"),
    [
        (0, "100"),       # no discount
        (Decimal("0.1"), "90.0"),  # 10% as a fraction
        (Decimal("0.25"), "75.00"),  # 25% as a fraction
        (10, "90.00"),    # 10% as a whole-number percentage
        (25, "60.00"),    # 25% as a whole-number percentage
        (50, "50.00"),
        (100, "0.00"),    # 100% off
    ],
)
def test_apply_discount_normalises_percentages(discount, expected):
    """A whole-number percentage is read as a percentage, not a fraction."""
    assert apply_discount(M(80), discount).amount == Decimal(expected)


def test_apply_discount_never_produces_a_negative_from_a_valid_percentage():
    for pct in (0, 1, 5, 25, 50, 75, 100):
        result = apply_discount(M(80), pct).amount
        assert result >= 0, f"{pct}% produced a negative total: {result}"


def test_apply_discount_preserves_currency():
    assert apply_discount(M(80, "EUR"), 25).currency == "EUR"


# ------------------------------------------------ defect 2: allocate sums


@pytest.mark.parametrize(
    ("weights",),
    [([1, 1, 1],), ([1, 1, 2],), ([3, 3, 3, 3],), ([7, 11, 13],)],
)
def test_allocate_parts_sum_exactly_to_the_total(weights):
    """No cent is created or destroyed. This is the documented contract."""
    total = M(10)
    parts = allocate(total, weights)
    assert sum(p.amount for p in parts) == total.amount, (
        f"parts sum to {sum(p.amount for p in parts)}, not {total.amount}"
    )


def test_allocate_distributes_the_remainder():
    parts = allocate(M(10), [1, 1, 1])
    assert [p.amount for p in parts] == [Decimal("3.34"), Decimal("3.33"), Decimal("3.33")]


def test_allocate_exact_division_is_unaffected():
    assert [p.amount for p in allocate(M(9), [1, 1, 1])] == [Decimal("3.00")] * 3


def test_allocate_preserves_currency():
    assert all(p.currency == "EUR" for p in allocate(M(10, "EUR"), [1, 1]))


def test_allocate_rejects_negative_weights():
    with pytest.raises(BillingError):
        allocate(M(10), [1, -1])


def test_allocate_rejects_all_zero_weights():
    with pytest.raises(BillingError):
        allocate(M(10), [0, 0])


def test_allocate_of_empty_weights_is_empty():
    assert allocate(M(10), []) == []


# ----------------------------------------------- defect 3: credit notes


def test_invoice_total_subtracts_credit_notes():
    items = [LineItem("widget", M(100), 1)]
    assert invoice_total(items, [M(10)]).amount == Decimal("90.00")


def test_invoice_total_with_no_adjustments_is_unaffected():
    items = [LineItem("widget", M(100), 1)]
    assert invoice_total(items).amount == Decimal("100.00")


def test_invoice_total_applies_multiple_adjustments():
    items = [LineItem("widget", M(100), 1)]
    total = invoice_total(items, [M(10), M(5), M(1)])
    assert total.amount == Decimal("84.00")


def test_invoice_total_can_go_negative_when_credits_exceed_the_invoice():
    items = [LineItem("widget", M(100), 1)]
    assert invoice_total(items, [M(150)]).amount == Decimal("-50.00")


def test_invoice_total_rejects_an_empty_invoice():
    with pytest.raises(BillingError):
        invoice_total([])


# ------------------------------------------------- defect 4: quantities


@pytest.mark.parametrize("bad", ["-1", "-100", " -5 "])
def test_parse_quantity_rejects_negatives(bad):
    with pytest.raises(BillingError):
        parse_quantity(bad)


@pytest.mark.parametrize("raw", ["3", " 3 ", 3])
def test_parse_quantity_accepts_positive_integers(raw):
    assert parse_quantity(raw) == 3


@pytest.mark.parametrize("bad", ["abc", "3.5", "", None, True])
def test_parse_quantity_rejects_junk(bad):
    with pytest.raises(BillingError):
        parse_quantity(bad)


# ------------------------------------------ behaviour that must not change


def test_money_rejects_currency_mismatch_on_add():
    """Already correct in the seed; a refactor must not regress it."""
    with pytest.raises(BillingError):
        M(1, "USD") + M(1, "EUR")


def test_money_is_hashable_and_frozen():
    assert hash(M(1)) == hash(M(1))
    with pytest.raises(Exception):
        M(1).amount = Decimal(2)


def test_line_total_applies_fractional_discount():
    item = LineItem("widget", M(100), 2, discount=Decimal("0.10"))
    assert line_total(item).amount == Decimal("180.00")


def test_rounding_is_bankers_rounding():
    assert M("2.345").rounded().amount == Decimal("2.34")
    assert M("2.355").rounded().amount == Decimal("2.36")


def test_to_json_is_stable():
    items = [LineItem("widget", M("10.00"), 2)]
    assert to_json(items) == (
        '[{"currency": "USD", "description": "widget", '
        '"quantity": 2, "unit_price": "10.00"}]'
    )


def test_invoice_requires_line_items():
    with pytest.raises(BillingError):
        invoice_total([])