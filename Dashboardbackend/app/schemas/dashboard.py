from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class StatsQueries(BaseModel):
    open: int
    rates_available: int
    rates_quoted: int
    rates_confirmed: int
    lost: int

class StatsBookings(BaseModel):
    booking: int
    pending: int
    pricing_approval: int

class StatsShipment(BaseModel):
    created: int
    final: int
    executed: int

class StatsInvoice(BaseModel):
    total: int

class StatsWeight(BaseModel):
    charge: float
    gross: float

class DailyStats(BaseModel):
    date: datetime
    queries: StatsQueries
    bookings: StatsBookings
    shipment: StatsShipment
    invoice: StatsInvoice
    weight: StatsWeight
    byqueryFor: Optional[Dict[str, 'DailyStatsByqueryFor']] = None
    byQueryType: Optional[Dict[str, 'DailyStatsByqueryFor']] = None

class DailyStatsByqueryFor(BaseModel):
    queries: StatsQueries
    bookings: StatsBookings
    shipment: StatsShipment
    invoice: StatsInvoice
    weight: StatsWeight

class DashboardResponse(BaseModel):
    queries: StatsQueries
    bookings: StatsBookings
    shipment: StatsShipment
    invoice: StatsInvoice
    weight: StatsWeight
    history: List[DailyStats]
    byqueryFor: Dict[str, DailyStatsByqueryFor]
    byQueryType: Dict[str, DailyStatsByqueryFor]
    byClient: Dict[str, int]
    byRoute: Dict[str, int]

# Forward reference for recursive definition
DailyStats.model_rebuild()

class DashboardWidgetConfig(BaseModel):
    title: str
    description: Optional[str] = None
    icon: Optional[str] = None
    value: Optional[str] = None
    chartType: Optional[str] = None
    dataKey: Optional[str] = None
    shipmentType: Optional[str] = None
    queryType: Optional[str] = None
    range: Optional[str] = None
    groupBy: Optional[str] = None
    orientation: Optional[str] = None
    colSpan: Optional[int] = None
    rowSpan: Optional[int] = None
    class Config:
        extra = "allow" # allow pivotConfig, subcategories etc without strict typing

class DashboardWidget(BaseModel):
    id: str
    type: str
    config: DashboardWidgetConfig

class DashboardTemplateCreate(BaseModel):
    name: str
    widgets: List[DashboardWidget]
    assigned_departments: Optional[List[str]] = []
    assigned_roles: Optional[List[str]] = []

class DashboardTemplateInDB(DashboardTemplateCreate):
    id: str = Field(alias="_id")
    created_at: datetime
    created_by: str

    class Config:
        populate_by_name = True
