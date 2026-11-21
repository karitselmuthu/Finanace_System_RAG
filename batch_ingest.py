#!/usr/bin/env python3
"""Batch ingestion script — processes all PDF files in the raw PDF directory.

Walks through every *.pdf in the configured (or supplied) directory, skips
files that have already been ingested, and runs the full pipeline on the rest:

    load → preprocess → chunk → embed (ChromaDB) → parse transactions → save JSON

Usage
-----
    python batch_ingest.py                   # process data/raw/ (skip already done)
    python batch_ingest.py --force           # re-process everything
    python batch_ingest.py --dir path/to/   # custom PDF directory
    python batch_ingest.py --dry-run        # preview only, no changes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure the project root is on sys.path so src.* imports resolve when the
# script is executed directly (i.e. not as part of an installed package).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.embedding.vector_store import upsert_chunks
from src.extraction.transaction_parser import parse_transactions
from src.ingestion.chunker import chunk_document
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.preprocessor import preprocess_document
from src.utils.config import settings
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FileResult:
    filename: str
    doc_id: str = ""
    status: str = "pending"   # pending | skipped | success | failed | dry-run
    pages: int = 0
    chunks_stored: int = 0
    transactions_found: int = 0
    elapsed: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _quick_doc_id(pdf_path: Path) -> str:
    """Compute doc_id using the same SHA-256 logic as pdf_loader.load_pdf."""
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]


def _already_processed(doc_id: str, processed_dir: Path) -> bool:
    return (processed_dir / f"{doc_id}.json").exists()


def process_pdf(pdf_path: Path, processed_dir: Path) -> FileResult:
    """Run the full ingestion pipeline for a single PDF.

    Returns a FileResult whose ``status`` is either ``"success"`` or
    ``"failed"``.
    """
    result = FileResult(filename=pdf_path.name)
    start = time.perf_counter()
    try:
        # 1. Load
        doc = load_pdf(pdf_path)
        result.doc_id = doc.doc_id
        result.pages = doc.total_pages

        # 2. Preprocess
        doc = preprocess_document(doc)

        # 3. Chunk
        chunks = chunk_document(doc)

        # 4. Embed + store in ChromaDB
        upsert_chunks(chunks)
        result.chunks_stored = len(chunks)

        # 5. Extract structured transactions
        transactions = parse_transactions(doc)
        result.transactions_found = len(transactions)

        # 6. Persist transactions as JSON
        txn_dicts = [t.model_dump(mode="json") for t in transactions]
        out_path = processed_dir / f"{doc.doc_id}.json"
        out_path.write_text(json.dumps(txn_dicts, indent=2, default=str), encoding="utf-8")

        result.status = "success"

    except Exception as exc:
        result.status = "failed"
        result.error = str(exc)
        logger.error(f"Failed to process {pdf_path.name}: {exc}")

    finally:
        result.elapsed = time.perf_counter() - start

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-ingest PDF bank statements into the RAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dir",
        metavar="PATH",
        default=None,
        help=f"Directory containing PDF files (default: {settings.raw_pdf_dir})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process PDFs that were already ingested (default: skip them)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List PDFs that would be processed without actually running the pipeline",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    pdf_dir = Path(args.dir) if args.dir else Path(settings.raw_pdf_dir)
    processed_dir = Path(settings.processed_dir)

    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        sys.exit(1)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"No PDF files found in {pdf_dir}")
        sys.exit(0)

    logger.info(f"Batch job started — {len(pdfs)} PDF(s) found in {pdf_dir}")
    if args.dry_run:
        logger.info("DRY-RUN mode: no files will be modified")
    if args.force:
        logger.info("FORCE mode: previously processed files will be re-ingested")

    results: list[FileResult] = []
    batch_start = time.perf_counter()

    for idx, pdf_path in enumerate(pdfs, start=1):
        prefix = f"[{idx}/{len(pdfs)}]"

        # Pre-compute doc_id so we can detect duplicates before loading the PDF
        doc_id = _quick_doc_id(pdf_path)

        # Skip check (unless --force)
        if not args.force and _already_processed(doc_id, processed_dir):
            logger.info(f"{prefix} SKIP  {pdf_path.name}  (doc_id={doc_id} already in {processed_dir})")
            results.append(FileResult(filename=pdf_path.name, doc_id=doc_id, status="skipped"))
            continue

        # Dry-run: just report what would happen
        if args.dry_run:
            logger.info(f"{prefix} WOULD PROCESS  {pdf_path.name}  (doc_id={doc_id})")
            results.append(FileResult(filename=pdf_path.name, doc_id=doc_id, status="dry-run"))
            continue

        # Process
        logger.info(f"{prefix} Processing  {pdf_path.name}  …")
        result = process_pdf(pdf_path, processed_dir)
        results.append(result)

        if result.status == "success":
            logger.info(
                f"{prefix} OK    {pdf_path.name} — "
                f"pages={result.pages}, "
                f"chunks={result.chunks_stored}, "
                f"transactions={result.transactions_found}, "
                f"time={result.elapsed:.1f}s"
            )
        else:
            logger.error(f"{prefix} FAIL  {pdf_path.name} — {result.error}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - batch_start

    success = [r for r in results if r.status == "success"]
    skipped = [r for r in results if r.status == "skipped"]
    failed  = [r for r in results if r.status == "failed"]
    dry_run = [r for r in results if r.status == "dry-run"]

    sep = "=" * 60
    print(f"\n{sep}")
    print("BATCH INGESTION SUMMARY")
    print(sep)
    print(f"  PDF directory      : {pdf_dir}")
    print(f"  Total PDFs found   : {len(pdfs)}")
    print(f"    Processed (new)  : {len(success)}")
    print(f"    Skipped (done)   : {len(skipped)}")
    print(f"    Failed           : {len(failed)}")
    if dry_run:
        print(f"    Dry-run preview  : {len(dry_run)}")
    print(f"  Chunks stored      : {sum(r.chunks_stored for r in success)}")
    print(f"  Transactions found : {sum(r.transactions_found for r in success)}")
    print(f"  Total time         : {total_elapsed:.1f}s")

    if success:
        print("\n  Processed files:")
        for r in success:
            print(
                f"    {r.filename:<40}  doc_id={r.doc_id}  "
                f"pages={r.pages}  chunks={r.chunks_stored}  "
                f"txns={r.transactions_found}  ({r.elapsed:.1f}s)"
            )

    if failed:
        print("\n  Failed files:")
        for r in failed:
            print(f"    {r.filename:<40}  {r.error}")

    print(sep)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
