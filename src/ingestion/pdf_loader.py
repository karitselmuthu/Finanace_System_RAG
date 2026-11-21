from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from src.utils.logger import logger


@dataclass
class PageContent:
    page_number: int
    text: str
    tables: list[list[list[str | None]]]  # list of tables; each table is rows of cells


@dataclass
class PDFDocument:
    file_path: str
    doc_id: str
    total_pages: int
    pages: list[PageContent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _compute_doc_id(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]


def load_pdf(file_path: str | Path) -> PDFDocument:
    """Extract text and tables from a PDF file using pdfplumber."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc_id = _compute_doc_id(path)
    logger.info(f"Loading PDF: {path.name}  (doc_id={doc_id})")

    pages: list[PageContent] = []

    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            # Normalize table cells to str
            clean_tables = [
                [[cell if cell is not None else "" for cell in row] for row in table]
                for table in tables
            ]
            pages.append(
                PageContent(
                    page_number=page.page_number,
                    text=text.strip(),
                    tables=clean_tables,
                )
            )
            logger.debug(
                f"  Page {page.page_number}: {len(text)} chars, {len(tables)} tables"
            )

    return PDFDocument(
        file_path=str(path),
        doc_id=doc_id,
        total_pages=total_pages,
        pages=pages,
        metadata={"filename": path.name, "size_bytes": path.stat().st_size},
    )
