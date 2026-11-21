from __future__ import annotations

import io
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ingestion.pdf_loader import load_pdf, PDFDocument, PageContent


class TestLoadPDF:
    def test_raises_when_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pdf(tmp_path / "missing.pdf")

    def test_returns_pdf_document(self, tmp_path):
        """Smoke test: mock pdfplumber and verify PDFDocument is returned."""
        fake_pdf = tmp_path / "statement.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake content for hashing")

        mock_page = MagicMock()
        mock_page.page_number = 1
        mock_page.extract_text.return_value = "Date  Description  Amount\n2026-01-15  Coffee  5.00"
        mock_page.extract_tables.return_value = []

        mock_pdf_ctx = MagicMock()
        mock_pdf_ctx.__enter__ = MagicMock(return_value=mock_pdf_ctx)
        mock_pdf_ctx.__exit__ = MagicMock(return_value=False)
        mock_pdf_ctx.pages = [mock_page]

        with patch("pdfplumber.open", return_value=mock_pdf_ctx):
            doc = load_pdf(fake_pdf)

        assert isinstance(doc, PDFDocument)
        assert doc.total_pages == 1
        assert len(doc.pages) == 1
        assert "Coffee" in doc.pages[0].text

    def test_doc_id_is_deterministic(self, tmp_path):
        fake_pdf = tmp_path / "statement.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 stable bytes")

        mock_page = MagicMock()
        mock_page.page_number = 1
        mock_page.extract_text.return_value = "text"
        mock_page.extract_tables.return_value = []
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.pages = [mock_page]

        with patch("pdfplumber.open", return_value=mock_ctx):
            doc1 = load_pdf(fake_pdf)
            doc2 = load_pdf(fake_pdf)

        assert doc1.doc_id == doc2.doc_id
