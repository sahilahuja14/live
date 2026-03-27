from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class FinanceSummary(BaseModel):
    Turnover: float = Field(0.0, description="Total taxable amount")
    Margin: float = Field(0.0, description="Profit amount (Turnover - Buy)")
    Tonnage: float = Field(0.0, description="Chargeable weight in Metric Tons")
    Buy: float = Field(0.0, description="Total cost amount")
    Count: int = Field(0, description="Number of invoices")

class FinanceHistoryItem(FinanceSummary):
    date: str = Field(..., description="Date string in YYYY-MM-DD format")

class FinanceGranularItem(FinanceSummary):
    date: str = Field(..., description="Date string in YYYY-MM-DD format")
    mode: str = Field(..., description="Transport mode")
    customer: str = Field(..., description="Customer Name")
    route: str = Field(..., description="Origin -> Destination route")
    carrier: str = Field(..., description="Carrier Name")

class FinanceRiskMetrics(BaseModel):
    low_margin: int = Field(0, description="Count of low margin accounts (0-5%)")
    loss_making: int = Field(0, description="Count of loss making accounts (<0)")

class FinanceStatsResponse(BaseModel):
    summary: FinanceSummary
    history: List[FinanceHistoryItem]
    byCustomer: Dict[str, FinanceSummary]
    byRoute: Dict[str, FinanceSummary]
    byCarrier: Dict[str, FinanceSummary]
    granular: List[FinanceGranularItem]
    risk: FinanceRiskMetrics
