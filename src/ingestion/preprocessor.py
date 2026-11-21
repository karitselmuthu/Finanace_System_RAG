from __future__ import annotations

import re

from src.ingestion.pdf_loader import PDFDocument, PageContent
from src.utils.logger import logger

# Patterns for common noise in bank statements
_NOISE_PATTERNS = [
    re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"Continued\s+on\s+next\s+page", re.IGNORECASE),
    re.compile(r"^\s*[-–—]+\s*$", re.MULTILINE),   # horizontal rules
    re.compile(r"\x0c"),                             # form-feed chars
    re.compile(r"[ \t]{2,}"),                        # collapse whitespace
]

# Normalize common date formats to ISO 8601
_DATE_NORMALIZATION = [
    # DD/MM/YYYY  or  DD-MM-YYYY  →  YYYY-MM-DD
    (re.compile(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b"), r"\3-\2-\1"),
    # Month DD, YYYY  →  YYYY-Month-DD (kept as text, parser handles it)
]

# Normalize currency symbols to a plain token so embeddings are consistent
_CURRENCY_MAP = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}


def _clean_text(text: str) -> str:
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub(" ", text)

    for pattern, repl in _DATE_NORMALIZATION:
        text = pattern.sub(repl, text)

    for symbol, code in _CURRENCY_MAP.items():
        text = text.replace(symbol, f"{code} ")

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _table_to_text(table: list[list[str]]) -> str:
    """Convert a table (list of rows) to a readable plain-text block."""
    if not table:
        return ""
    lines = []
    for row in table:
        lines.append("  |  ".join(cell.strip() for cell in row if cell.strip()))
    return "\n".join(lines)


def preprocess_document(doc: PDFDocument) -> PDFDocument:
    """
    Clean every page's text and inline table content.
    Returns the same PDFDocument with cleaned text fields.
    """
    logger.info(f"Preprocessing document: {doc.metadata.get('filename')}")
    for page in doc.pages:
        page.text = _clean_text(page.text)

        # Append table text to the page text so it's searchable
        for table in page.tables:
            table_text = _table_to_text(table)
            if table_text:
                page.text += "\n\n[TABLE]\n" + table_text + "\n[/TABLE]"

    return doc
