from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel

from src.extraction.schema import TransactionType, TransactionStatus


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    total_pages: int
    chunks_stored: int
    transactions_found: int
    message: str = "PDF ingested successfully"


class QueryRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None       # restrict to a specific document
    top_k: Optional[int] = None


class TransactionOut(BaseModel):
    id: str
    transaction_date: date
    description: str
    amount: Decimal
    transaction_type: TransactionType
    status: TransactionStatus
    balance: Optional[Decimal] = None
    reference: Optional[str] = None
    currency: str = "USD"
    source_doc: Optional[str] = None
    source_page: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    transactions: list[TransactionOut] = []
    sources: list[str] = []
    query: str


class TransactionListResponse(BaseModel):
    transactions: list[TransactionOut]
    total: int
    credits: int
    debits: int


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
