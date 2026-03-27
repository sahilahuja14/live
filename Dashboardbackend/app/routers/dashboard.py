import fastapi
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from datetime import datetime
from ..schemas.dashboard import DashboardResponse, DashboardTemplateCreate, DashboardTemplateInDB
from ..services.dashboard_stats import calculate_dashboard_stats
from ..core.deps import get_current_user
from ..core.permissions import get_permissions_for_role
from ..schemas.auth import UserInDB
from ..database import get_async_database
from bson import ObjectId

router = APIRouter()

@router.get("/stats", response_model=DashboardResponse)
async def get_dashboard_stats(
    range_param: Optional[str] = Query(None, alias="range"),
    from_date: Optional[datetime] = Query(None, alias="from"),
    to_date: Optional[datetime] = Query(None, alias="to"),
    shipment_type: str = Query("all", alias="mode"),
    query_type: Optional[str] = Query(None, alias="queryType")
):
    stats = await calculate_dashboard_stats(
        range_param=range_param,
        from_date=from_date,
        to_date=to_date,
        shipment_type=shipment_type,
        query_type=query_type
    )
    return stats

@router.get("/templates", response_model=list[DashboardTemplateInDB])
async def get_templates(current_user: UserInDB = fastapi.Depends(get_current_user)):
    db = get_async_database()
    
    # Superadmin sees all templates
    if current_user.role == "superadmin":
        cursor = db.templates.find()
    else:
        # Others see templates assigned to their role OR their department
        query_conditions = []
        if current_user.role:
            query_conditions.append({"assigned_roles": current_user.role})
        if current_user.department_id:
            query_conditions.append({"assigned_departments": current_user.department_id})
        
        if not query_conditions:
            return [] # No role or department, no templates
            
        cursor = db.templates.find({"$or": query_conditions})
        
    templates = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        templates.append(DashboardTemplateInDB(**doc))
    return templates

@router.post("/templates", response_model=DashboardTemplateInDB)
async def create_template(
    template_in: DashboardTemplateCreate,
    current_user: UserInDB = fastapi.Depends(get_current_user)
):
    db = get_async_database()
    
    template_data = template_in.model_dump()
    template_data["created_at"] = datetime.utcnow()
    template_data["created_by"] = current_user.username
    
    result = await db.templates.insert_one(template_data)
    
    created_template = await db.templates.find_one({"_id": result.inserted_id})
    created_template["_id"] = str(created_template["_id"])
    return DashboardTemplateInDB(**created_template)

@router.get("/permissions", response_model=list[str])
async def get_permissions(current_user: UserInDB = fastapi.Depends(get_current_user)):
    """Returns the allowed widget data keys for the user's role."""
    return get_permissions_for_role(current_user.role or "user")

@router.delete("/templates/{template_id}")
async def delete_template(
    template_id: str,
    current_user: UserInDB = fastapi.Depends(get_current_user)
):
    if current_user.role != "superadmin":
        raise HTTPException(status_code=403, detail="Only superadmins can delete templates.")
    
    db = get_async_database()
    try:
        result = await db.templates.delete_one({"_id": ObjectId(template_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid template ID format.")
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Template not found.")
    
    return {"status": "ok", "deleted_id": template_id}
