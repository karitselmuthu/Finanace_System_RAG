from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.pdf_loader import PDFDocument
from src.utils.logger import logger

DEFAULT_CHUNK_SIZE = 800      # characters
DEFAULT_CHUNK_OVERLAP = 150   # characters


@dataclass
class Chunk:
    chunk_id: str           # "{doc_id}_p{page}_c{index}"
    doc_id: str
    page_number: int
    text: str
    metadata: dict


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Slide a window over text, splitting on whitespace boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Prefer splitting at a newline or space
            split_at = text.rfind("\n", start, end)
            if split_at == -1:
                split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def chunk_document(
    doc: PDFDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split every page in the document into overlapping text chunks.
    Each chunk carries metadata for source attribution.
    """
    logger.info(
        f"Chunking document {doc.doc_id} "
        f"(chunk_size={chunk_size}, overlap={overlap})"
    )
    chunks: list[Chunk] = []

    for page in doc.pages:
        if not page.text.strip():
            continue
        page_chunks = _split_text(page.text, chunk_size, overlap)
        for idx, text in enumerate(page_chunks):
            chunk_id = f"{doc.doc_id}_p{page.page_number}_c{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    page_number=page.page_number,
                    text=text,
                    metadata={
                        "doc_id": doc.doc_id,
                        "filename": doc.metadata.get("filename", ""),
                        "page_number": page.page_number,
                        "chunk_index": idx,
                    },
                )
            )

    logger.info(f"  → {len(chunks)} chunks produced")
    return chunks
