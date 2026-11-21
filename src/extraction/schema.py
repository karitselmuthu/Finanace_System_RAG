from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TransactionType(str, Enum):
    CREDIT = "credit"
    DEBIT = "debit"
    UNKNOWN = "unknown"


class TransactionStatus(str, Enum):
    COMPLETED = "completed"
    PENDING = "pending"
    FAILED = "failed"
    REVERSED = "reversed"
    UNKNOWN = "unknown"


class Transaction(BaseModel):
    id: str = Field(description="Unique transaction identifier")
    transaction_date: date = Field(description="Transaction date (ISO 8601)")
    description: str = Field(description="Merchant or transaction description")
    amount: Decimal = Field(ge=0, description="Absolute transaction amount")
    transaction_type: TransactionType = Field(description="credit or debit")
    status: TransactionStatus = Field(default=TransactionStatus.COMPLETED)
    balance: Optional[Decimal] = Field(default=None, description="Running balance after transaction")
    reference: Optional[str] = Field(default=None, description="Bank reference / cheque number")
    currency: str = Field(default="USD")
    source_doc: Optional[str] = Field(default=None, description="Source PDF filename")
    source_page: Optional[int] = Field(default=None, description="Page number in source PDF")

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_amount(cls, v):
        if isinstance(v, str):
            v = v.replace(",", "").replace(" ", "")
        return Decimal(str(v))


class TransactionSummary(BaseModel):
    total_credits: Decimal = Field(default_factory=lambda: Decimal("0"))
    total_debits: Decimal = Field(default_factory=lambda: Decimal("0"))
    net_balance: Decimal = Field(default_factory=lambda: Decimal("0"))
    transaction_count: int = 0
    credit_count: int = 0
    debit_count: int = 0
