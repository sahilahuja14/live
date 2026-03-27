from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from datetime import datetime
from ..services.dashboard_stats import calculate_financial_stats
from ..schemas.finance import FinanceStatsResponse

router = APIRouter()

@router.get("/stats", response_model=FinanceStatsResponse)
async def get_finance_stats(
    range: Optional[str] = Query("monthly", description="Time range preset (daily, weekly, monthly, quarterly, yearly, all)"),
    from_date: Optional[datetime] = Query(None, description="Start date for custom range"),
    to_date: Optional[datetime] = Query(None, description="End date for custom range"),
    modes: Optional[List[str]] = Query(None, description="Filter by transport modes")
):
    """
    Fetch financial statistics including turnover, margin, and tonnage.
    """
    try:
        stats = await calculate_financial_stats(
            range_param=range,
            from_date=from_date,
            to_date=to_date,
            modes=modes
        )
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
