from __future__ import annotations

import pytest
from datetime import date
from decimal import Decimal

from src.extraction.schema import TransactionType, TransactionStatus
from src.extraction.transaction_parser import (
    _parse_date,
    _parse_amount,
    _infer_type,
    _infer_status,
)
from src.extraction.classifier import summarize, filter_by_type, filter_by_date_range
from src.extraction.schema import Transaction


# ── Helper to build a Transaction quickly ─────────────────────────────────────

def make_txn(
    txn_type=TransactionType.DEBIT,
    amount="100.00",
    status=TransactionStatus.COMPLETED,
    txn_date="2026-01-15",
):
    return Transaction(
        id="txn_test",
        transaction_date=date.fromisoformat(txn_date),
        description="Test",
        amount=Decimal(amount),
        transaction_type=txn_type,
        status=status,
    )


# ── Date parsing ───────────────────────────────────────────────────────────────

class TestParseDates:
    def test_iso_format(self):
        assert _parse_date("2026-01-15") == date(2026, 1, 15)

    def test_slash_format(self):
        assert _parse_date("15/01/2026") == date(2026, 1, 15)

    def test_dash_format(self):
        assert _parse_date("15-01-2026") == date(2026, 1, 15)

    def test_returns_none_for_no_date(self):
        assert _parse_date("no date here") is None

    def test_embedded_in_text(self):
        result = _parse_date("Transaction on 2026-03-01 for goods")
        assert result == date(2026, 3, 1)


# ── Amount parsing ─────────────────────────────────────────────────────────────

class TestParseAmounts:
    def test_simple_amount(self):
        assert _parse_amount("Payment 120.50") == Decimal("120.50")

    def test_comma_formatted(self):
        assert _parse_amount("USD 1,234.00") == Decimal("1234.00")

    def test_returns_none_when_no_amount(self):
        assert _parse_amount("no numbers here") is None


# ── Type inference ─────────────────────────────────────────────────────────────

class TestInferType:
    def test_credit_keyword(self):
        assert _infer_type("Credit received from ABC") == TransactionType.CREDIT

    def test_debit_keyword(self):
        assert _infer_type("Debit card payment") == TransactionType.DEBIT

    def test_deposit(self):
        assert _infer_type("Deposit 500.00") == TransactionType.CREDIT

    def test_withdrawal(self):
        assert _infer_type("ATM withdrawal 200.00") == TransactionType.DEBIT

    def test_unknown(self):
        assert _infer_type("some unrelated text") == TransactionType.UNKNOWN


# ── Status inference ───────────────────────────────────────────────────────────

class TestInferStatus:
    def test_failed(self):
        assert _infer_status("Payment failed") == TransactionStatus.FAILED

    def test_pending(self):
        assert _infer_status("Transaction pending approval") == TransactionStatus.PENDING

    def test_reversed(self):
        assert _infer_status("Reversal processed") == TransactionStatus.REVERSED

    def test_default_completed(self):
        assert _infer_status("Grocery store 45.00") == TransactionStatus.COMPLETED


# ── Classifier ────────────────────────────────────────────────────────────────

class TestClassifier:
    def test_summarize(self):
        txns = [
            make_txn(TransactionType.CREDIT, "500.00"),
            make_txn(TransactionType.DEBIT, "200.00"),
            make_txn(TransactionType.DEBIT, "50.00"),
        ]
        summary = summarize(txns)
        assert summary.total_credits == Decimal("500.00")
        assert summary.total_debits == Decimal("250.00")
        assert summary.net_balance == Decimal("250.00")
        assert summary.credit_count == 1
        assert summary.debit_count == 2

    def test_filter_by_type(self):
        txns = [
            make_txn(TransactionType.CREDIT),
            make_txn(TransactionType.DEBIT),
            make_txn(TransactionType.DEBIT),
        ]
        assert len(filter_by_type(txns, TransactionType.DEBIT)) == 2
        assert len(filter_by_type(txns, TransactionType.CREDIT)) == 1

    def test_filter_by_date_range(self):
        txns = [
            make_txn(txn_date="2026-01-05"),
            make_txn(txn_date="2026-01-15"),
            make_txn(txn_date="2026-02-01"),
        ]
        result = filter_by_date_range(txns, start="2026-01-10", end="2026-01-31")
        assert len(result) == 1
        assert result[0].transaction_date == date(2026, 1, 15)
