from __future__ import annotations

import json
import traceback
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.models import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    TransactionListResponse,
    TransactionOut,
)
from src.embedding.vector_store import upsert_chunks
from src.extraction.classifier import classify_transactions, summarize
from src.extraction.transaction_parser import parse_transactions
from src.generation.llm_client import answer_query, answer_with_tools
from src.ingestion.chunker import chunk_document
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.preprocessor import preprocess_document
from src.retrieval.reranker import rerank
from src.retrieval.retriever import retrieve
from src.utils.config import settings
from src.utils.logger import logger

router = APIRouter()

# In-memory store for extracted transactions (keyed by doc_id).
# Replace with a proper DB (SQLite / Postgres) for production.
_transaction_store: dict[str, list] = {}
# Reload previously extracted transactions so they survive restarts.
for _f in Path(settings.processed_dir).glob("*.json"):
    try:
        _transaction_store[_f.stem] = json.loads(_f.read_text())
    except Exception:
        pass


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    return HealthResponse()


@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload a PDF statement and process it end-to-end."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    try:
        # Save upload
        save_path = Path(settings.raw_pdf_dir) / file.filename
        save_path.write_bytes(await file.read())
        logger.info(f"Saved upload: {save_path}")

        # Ingestion pipeline
        doc = load_pdf(save_path)
        doc = preprocess_document(doc)
        chunks = chunk_document(doc)

        # Embed + store vectors
        logger.info("Embedding and storing chunks...")
        upsert_chunks(chunks)
        logger.info(f"Stored {len(chunks)} chunks in vector DB")

        # Extract transactions
        transactions = parse_transactions(doc)
        _transaction_store[doc.doc_id] = [t.model_dump(mode="json") for t in transactions]

        # Persist extracted transactions as JSON
        out_path = Path(settings.processed_dir) / f"{doc.doc_id}.json"
        out_path.write_text(
            json.dumps(_transaction_store[doc.doc_id], indent=2, default=str)
        )

        return IngestResponse(
            doc_id=doc.doc_id,
            filename=file.filename,
            total_pages=doc.total_pages,
            chunks_stored=len(chunks),
            transactions_found=len(transactions),
        )

    except Exception as e:
        logger.error(f"Ingest failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {type(e).__name__}: {e}")


@router.post("/query", response_model=QueryResponse, tags=["query"])
async def query_transactions(body: QueryRequest):
    """Ask a natural-language question about transaction data."""
    try:
        # Structured path: exact calculations over extracted transactions.
        rows = (
            _transaction_store.get(body.doc_id, [])
            if body.doc_id
            else [t for ts in _transaction_store.values() for t in ts]
        )
        if rows:
            result = answer_with_tools(body.query, rows)
            safe = []
            for raw in result["transactions"]:
                try:
                    safe.append(TransactionOut(**raw))
                except Exception:
                    pass
            return QueryResponse(
                answer=result["answer"],
                transactions=safe,
                sources=sorted({r.get("source_doc") or "unknown" for r in rows}),
                query=body.query,
            )

        # Fallback: no structured data (e.g. unparsed layout) -> RAG over text chunks.
        hits = retrieve(body.query, top_k=body.top_k, doc_id=body.doc_id)
        if not hits:
            raise HTTPException(
                status_code=404,
                detail="No relevant context found. Try ingesting a PDF first.",
            )

        hits = rerank(hits, body.query, top_n=5)
        result = answer_query(body.query, hits)

        sources = list(
            {
                f"{h['metadata'].get('filename', 'unknown')} p.{h['metadata'].get('page_number', '?')}"
                for h in hits
            }
        )

        # Try to coerce LLM-returned transactions into TransactionOut
        safe_txns: list[TransactionOut] = []
        for raw in result.get("transactions", []):
            try:
                safe_txns.append(TransactionOut(**raw))
            except Exception:
                pass

        return QueryResponse(
            answer=result["answer"],
            transactions=safe_txns,
            sources=sources,
            query=body.query,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Query failed: {type(e).__name__}: {e}")


@router.get("/transactions", response_model=TransactionListResponse, tags=["transactions"])
async def list_transactions(doc_id: str | None = None):
    """Return all extracted transactions, optionally filtered by document."""
    raw: list[dict] = []
    if doc_id:
        raw = _transaction_store.get(doc_id, [])
    else:
        for txns in _transaction_store.values():
            raw.extend(txns)

    transactions = [TransactionOut(**t) for t in raw]
    credits = sum(1 for t in transactions if t.transaction_type.value == "credit")
    debits = sum(1 for t in transactions if t.transaction_type.value == "debit")

    return TransactionListResponse(
        transactions=transactions,
        total=len(transactions),
        credits=credits,
        debits=debits,
    )


@router.get(
    "/transactions/{transaction_id}",
    response_model=TransactionOut,
    tags=["transactions"],
)
async def get_transaction(transaction_id: str):
    """Retrieve a single transaction by ID."""
    for txns in _transaction_store.values():
        for t in txns:
            if t.get("id") == transaction_id:
                return TransactionOut(**t)
    raise HTTPException(status_code=404, detail=f"Transaction '{transaction_id}' not found")
