from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from src.extraction.schema import Transaction, TransactionType, TransactionStatus
from src.ingestion.pdf_loader import PDFDocument
from src.utils.logger import logger

# ── Regex patterns ─────────────────────────────────────────────────────────────

# ISO date:  2026-01-15
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
# Short date: 15/01/2026  or  01-15-2026
_SHORT_DATE = re.compile(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b")

# Amounts like:  1,234.56   or  1234.56
_AMOUNT = re.compile(r"[\d,]+\.\d{2}")

# Keywords that hint at credit / debit
_CREDIT_KEYWORDS = re.compile(
    r"\b(credit|deposit|received|refund|transfer in|cr)\b", re.IGNORECASE
)
_DEBIT_KEYWORDS = re.compile(
    r"\b(debit|withdrawal|payment|purchase|transfer out|dr)\b", re.IGNORECASE
)

# Status hints
_FAILED_KEYWORDS = re.compile(r"\b(failed|declined|rejected|error)\b", re.IGNORECASE)
_PENDING_KEYWORDS = re.compile(r"\b(pending|processing|hold)\b", re.IGNORECASE)
_REVERSED_KEYWORDS = re.compile(r"\b(reversed|reversal|cancelled|void)\b", re.IGNORECASE)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_date(text: str) -> date | None:
    m = _ISO_DATE.search(text)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass

    m = _SHORT_DATE.search(text)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(year, month, day)
        except ValueError:
            try:
                return date(year, day, month)  # try MM/DD/YYYY
            except ValueError:
                pass
    return None


def _parse_amount(text: str) -> Decimal | None:
    amounts = _AMOUNT.findall(text)
    if not amounts:
        return None
    # Take the first valid amount found
    for raw in amounts:
        try:
            return Decimal(raw.replace(",", ""))
        except InvalidOperation:
            continue
    return None


def _infer_type(text: str) -> TransactionType:
    if _CREDIT_KEYWORDS.search(text):
        return TransactionType.CREDIT
    if _DEBIT_KEYWORDS.search(text):
        return TransactionType.DEBIT
    return TransactionType.UNKNOWN


def _infer_status(text: str) -> TransactionStatus:
    if _FAILED_KEYWORDS.search(text):
        return TransactionStatus.FAILED
    if _PENDING_KEYWORDS.search(text):
        return TransactionStatus.PENDING
    if _REVERSED_KEYWORDS.search(text):
        return TransactionStatus.REVERSED
    return TransactionStatus.COMPLETED


def _make_id(doc_id: str, page: int, idx: int) -> str:
    raw = f"{doc_id}_p{page}_{idx}"
    return "txn_" + hashlib.md5(raw.encode()).hexdigest()[:10]


# ── Table-based parsing ────────────────────────────────────────────────────────

_TABLE_DATE_COL = re.compile(r"\bdate\b", re.IGNORECASE)
_TABLE_DESC_COL = re.compile(r"\b(desc|description|narration|particulars)\b", re.IGNORECASE)
_TABLE_DEBIT_COL = re.compile(r"\b(debit|dr|withdrawal|amount)\b", re.IGNORECASE)
_TABLE_CREDIT_COL = re.compile(r"\b(credit|cr|deposit)\b", re.IGNORECASE)
_TABLE_BALANCE_COL = re.compile(r"\bbalance\b", re.IGNORECASE)


def _parse_table(
    table: list[list[str]],
    doc_id: str,
    page: int,
    filename: str,
    start_idx: int,
) -> list[Transaction]:
    if len(table) < 2:
        return []

    header = [str(cell).strip() for cell in table[0]]

    def find_col(pattern: re.Pattern) -> int | None:
        for i, h in enumerate(header):
            if pattern.search(h):
                return i
        return None

    date_col = find_col(_TABLE_DATE_COL)
    desc_col = find_col(_TABLE_DESC_COL)
    debit_col = find_col(_TABLE_DEBIT_COL)
    credit_col = find_col(_TABLE_CREDIT_COL)
    balance_col = find_col(_TABLE_BALANCE_COL)

    transactions: list[Transaction] = []

    for row_idx, row in enumerate(table[1:], start=1):
        if not any(cell.strip() for cell in row):
            continue

        full_text = " ".join(str(c) for c in row)
        txn_date = _parse_date(row[date_col]) if date_col is not None else _parse_date(full_text)
        if txn_date is None:
            continue

        description = row[desc_col].strip() if desc_col is not None and desc_col < len(row) else full_text

        # Determine amount and type from dedicated columns
        txn_type = TransactionType.UNKNOWN
        amount: Decimal | None = None
        balance: Decimal | None = None

        if debit_col is not None and debit_col < len(row) and row[debit_col].strip():
            amount = _parse_amount(row[debit_col])
            if amount:
                txn_type = TransactionType.DEBIT

        if credit_col is not None and credit_col < len(row) and row[credit_col].strip():
            credit_amount = _parse_amount(row[credit_col])
            if credit_amount:
                amount = credit_amount
                txn_type = TransactionType.CREDIT

        if amount is None:
            amount = _parse_amount(full_text)
        if amount is None:
            continue

        if txn_type == TransactionType.UNKNOWN:
            txn_type = _infer_type(full_text)

        if balance_col is not None and balance_col < len(row):
            balance = _parse_amount(row[balance_col])

        transactions.append(
            Transaction(
                id=_make_id(doc_id, page, start_idx + row_idx),
                transaction_date=txn_date,
                description=description,
                amount=amount,
                transaction_type=txn_type,
                status=_infer_status(full_text),
                balance=balance,
                source_doc=filename,
                source_page=page,
            )
        )

    return transactions


# ── Text-line parsing ──────────────────────────────────────────────────────────

def _parse_text_lines(
    text: str,
    doc_id: str,
    page: int,
    filename: str,
    start_idx: int,
) -> list[Transaction]:
    transactions: list[Transaction] = []
    for idx, line in enumerate(text.splitlines()):
        line = line.strip()
        if len(line) < 10:
            continue
        txn_date = _parse_date(line)
        if txn_date is None:
            continue
        amount = _parse_amount(line)
        if amount is None:
            continue
        transactions.append(
            Transaction(
                id=_make_id(doc_id, page, start_idx + idx),
                transaction_date=txn_date,
                description=line,
                amount=amount,
                transaction_type=_infer_type(line),
                status=_infer_status(line),
                source_doc=filename,
                source_page=page,
            )
        )
    return transactions


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_transactions(doc: PDFDocument) -> list[Transaction]:
    """
    Extract all transactions from a preprocessed PDFDocument.
    Tries table parsing first; falls back to line-by-line text parsing.
    """
    all_transactions: list[Transaction] = []
    filename = doc.metadata.get("filename", "")
    counter = 0

    for page in doc.pages:
        page_transactions: list[Transaction] = []

        for table in page.tables:
            txns = _parse_table(table, doc.doc_id, page.page_number, filename, counter)
            page_transactions.extend(txns)
            counter += len(table)

        if not page_transactions:
            txns = _parse_text_lines(page.text, doc.doc_id, page.page_number, filename, counter)
            page_transactions.extend(txns)

        counter += len(page_transactions)
        all_transactions.extend(page_transactions)

    # Deduplicate by id
    seen: set[str] = set()
    unique = []
    for t in all_transactions:
        if t.id not in seen:
            seen.add(t.id)
            unique.append(t)

    logger.info(f"Extracted {len(unique)} transactions from {filename}")
    return unique
