from __future__ import annotations

from src.extraction.schema import Transaction, TransactionType, TransactionStatus, TransactionSummary
from src.utils.logger import logger


def summarize(transactions: list[Transaction]) -> TransactionSummary:
    """Aggregate totals for a list of transactions."""
    summary = TransactionSummary()
    for txn in transactions:
        summary.transaction_count += 1
        if txn.transaction_type == TransactionType.CREDIT:
            summary.total_credits += txn.amount
            summary.credit_count += 1
        elif txn.transaction_type == TransactionType.DEBIT:
            summary.total_debits += txn.amount
            summary.debit_count += 1
    summary.net_balance = summary.total_credits - summary.total_debits
    return summary


def filter_by_type(
    transactions: list[Transaction],
    txn_type: TransactionType,
) -> list[Transaction]:
    return [t for t in transactions if t.transaction_type == txn_type]


def filter_by_status(
    transactions: list[Transaction],
    status: TransactionStatus,
) -> list[Transaction]:
    return [t for t in transactions if t.status == status]


def filter_by_date_range(
    transactions: list[Transaction],
    start: str | None = None,
    end: str | None = None,
) -> list[Transaction]:
    """Filter by ISO date strings 'YYYY-MM-DD'. None means open bound."""
    from datetime import date

    def _parse(d: str) -> date:
        from datetime import datetime
        return datetime.strptime(d, "%Y-%m-%d").date()

    result = transactions
    if start:
        s = _parse(start)
        result = [t for t in result if t.transaction_date >= s]
    if end:
        e = _parse(end)
        result = [t for t in result if t.transaction_date <= e]
    return result


def classify_transactions(transactions: list[Transaction]) -> dict:
    """
    Returns a dict with:
        credits, debits, pending, failed, reversed — each a list of transactions.
    """
    return {
        "credits": filter_by_type(transactions, TransactionType.CREDIT),
        "debits": filter_by_type(transactions, TransactionType.DEBIT),
        "pending": filter_by_status(transactions, TransactionStatus.PENDING),
        "failed": filter_by_status(transactions, TransactionStatus.FAILED),
        "reversed": filter_by_status(transactions, TransactionStatus.REVERSED),
    }
