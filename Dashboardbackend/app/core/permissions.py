from typing import Dict, List

# Map user roles to the data keys they are allowed to see and add to their dashboards.
# 'all' means no restrictions.

ROLE_PERMISSIONS: Dict[str, List[str]] = {
    "superadmin": ["all"],
    
    # Sales is interested in queries and bookings
    "sales": ["queries", "bookings"],
    
    # Finance only cares about invoices and AWBs
    "finance": ["invoice", "awb"],
    
    # Operations cares about shipments/awb and bookings
    "operations": ["awb", "shipment", "bookings"]
}

def get_permissions_for_role(role: str) -> List[str]:
    """Retrieve the list of allowed data keys for a given role."""
    return ROLE_PERMISSIONS.get(role, [])
