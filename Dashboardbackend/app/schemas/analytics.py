from pydantic import BaseModel
from typing import List, Literal, Optional

class PivotField(BaseModel):
    id: str
    label: str
    type: str  # 'string', 'number', 'date'
    path: str

class PivotValue(BaseModel):
    fieldId: str
    aggregation: Literal['sum', 'count', 'avg', 'min', 'max']

class PivotConfig(BaseModel):
    filters: List[PivotField] = []
    rows: List[PivotField] = []
    columns: List[PivotField] = []
    values: List[PivotValue] = []
    limit: Optional[int] = None  # No limit by default (All Time)
