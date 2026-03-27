from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    customerId: str | None = None
    customerName: str | None = None
    commodity: str | None = None
    taxableTotalAmountB: float | None = Field(None, ge=0, le=1e12)
    incoTerms: str | None = None
    currency: str | None = None
    weight_discrepancy: float | None = Field(None, ge=0)
    invoiceNo: str | None = None
    invoiceDate: str | None = None
    executionDate: str | None = None
    dueDate: str | None = None
    paymentDate: str | None = None
    paymentDateRaw: str | None = None
    paidStatus: str | None = None
    termsDays: float | None = Field(None, ge=0, le=3650)
    grossAmount: float | None = None
    salesOwner: str | None = None
    documentType: str | None = None
    accountType: str | None = None
    company: str | None = None
    customer: dict[str, Any] | None = None
    shipmentDetails: dict[str, Any] | None = None
    operational: dict[str, Any] | None = None
    receivables: dict[str, Any] | None = None
    legacy: dict[str, Any] | None = None


class CustomerScoreRequest(BaseModel):
    customerId: str
    limit: int = Field(100, ge=1, le=5000)
    snapshotId: str | None = None
