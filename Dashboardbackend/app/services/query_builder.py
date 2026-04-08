from ..schemas.analytics import PivotConfig, PivotValue
from typing import List, Dict, Any

# Mapping Frontend IDs to Backend Paths
# Mapping Frontend IDs to Backend Paths
# Mapping Frontend IDs to Backend Paths
FIELD_MAPPING = {
    # Basic Mappings
    'status': 'quoteStatus',
    'shipmentType': 'queryFor',
    'origin': 'originAirport.code',
    'destination': 'destinationAirport.code',
    'clientName': 'customerName',
    'date': 'clearanceDate', # or createdAt
    
    # Metrics
    # Metrics
    'weight': 'chargeableWeight',         
    'volume': 'volumeWeight',        
    'grossWt': 'grossWeight',        
    'Pieces': 'totalPieces',   
    
    # Advanced / New Fields
    'jobId': 'jobNo',
    'activityType': 'activityType',
    'queryType': 'queryType',
    'queryFrom': 'queryFrom',
    'originAirport': 'originAirport.name',
    'originDoor': 'originDoor',
    'destinationAirport': 'destinationAirport.name',
    'destinationDoor': 'destinationDoor',
    'commodity': 'commodity',
    'hsn': 'hsn',

    'createdAt': 'createdAt',
    'updatedAt': 'updatedAt',
    'referenceNo': 'referenceNo',
    'referenceJobNo': 'referenceJobNo',
    'queryFor': 'queryFor',
    'jobNoPartNo': 'jobNoPartNo',
    'originName': 'originAirport.name',
    'destinationName': 'destinationAirport.name'
}

class QueryBuilder:
    def build_pipeline(self, config: PivotConfig) -> List[Dict[str, Any]]:
        pipeline = []
        
        # 0. Lookup / Join Stages
        # We join 'invoices' and 'bookings' on 'quoteId'
        # Assumption: 'queries' collection has 'quoteId' or 'id' implies 'quoteId'. 
        # User stated 'quoteId' is common. We will use 'quoteId' as the join key.
        # If 'queries' uses 'id' as the quoteId, we might need to adjust localField.
        # For now, we use 'quoteId' <-> 'quoteId'. 
        
        pipeline.extend([
            {
                "$lookup": {
                    "from": "invoices",
                    "localField": "_id", 
                    "foreignField": "quoteId",
                    "as": "invoice_docs"
                }
            },
            {
                "$lookup": {
                    "from": "bookings",
                    "localField": "_id",
                    "foreignField": "quoteId",
                    "as": "booking_docs"
                }
            },
            # Change unwind to preserve empty arrays so we don't lose queries without invoices/bookings
            {
                "$unwind": {
                    "path": "$invoice_docs",
                    "preserveNullAndEmptyArrays": True
                }
            },
             {
                "$unwind": {
                    "path": "$booking_docs",
                    "preserveNullAndEmptyArrays": True
                }
            }
        ])

        # 1. Selection & Projection Stage
        # We need to map Backend Keys -> Frontend IDs so the frontend logic finds the data.
        # We will use $project to rename fields and ensure types.
        
        project_stage = {
            "_id": 0, # Exclude ID unless needed
        }
        
        # Collect all fields needed (rows, cols, values, filters)
        # For simplicity, we can project ALL mapped fields, or just the ones in config.
        # Let's project based on config to be efficient, but also include basic ones if needed.
        
        # Helper to add field to project
        def add_project_field(frontend_id):
            # Handle Cross-Collection Fields
            if frontend_id.startswith("invoice."):
                part = frontend_id.split(".", 1)[1]
                
                # MAPPING: Frontend -> DB
                db_field = part
                if part == 'total': db_field = 'totalAmountB'
                elif part == 'amount': db_field = 'totalAmountB'
                
                # Special type handling for known fields
                if part in ['amount', 'total', 'tax'] or db_field in ['totalAmountB', 'totalAmountC']:
                     project_stage[frontend_id] = { 
                         "$convert": { 
                             "input": f"$invoice_docs.{db_field}", 
                             "to": "double", 
                             "onError": 0, 
                             "onNull": 0 
                         } 
                     }
                else:
                     project_stage[frontend_id] = { "$ifNull": [ f"$invoice_docs.{db_field}", None ] }
                return

            if frontend_id.startswith("booking."):
                part = frontend_id.split(".", 1)[1]
                
                # MAPPING
                db_field = part
                if part == 'status': db_field = 'blStatus'

                project_stage[frontend_id] = { "$ifNull": [ f"$booking_docs.{db_field}", None ] }
                return

            # Normal Fields
            backend_key = FIELD_MAPPING.get(frontend_id, frontend_id)
            
            # Type Conversion Logic
            if frontend_id == 'weight':
                 project_stage[frontend_id] = {
                     "$divide": [
                         {"$convert": { "input": f"${backend_key}", "to": "double", "onError": 0.0, "onNull": 0.0 }},
                         1000.0
                     ]
                 }
            elif frontend_id in ['volume', 'grossWt', 'amount']:
                 project_stage[frontend_id] = {
                     "$convert": { "input": f"${backend_key}", "to": "double", "onError": 0, "onNull": 0 }
                 }
            elif frontend_id == 'Pieces':
                 project_stage[frontend_id] = { "$toInt": f"${backend_key}" }
            elif frontend_id == 'date':
                 # Ensure date string? It is string in DB.
                 project_stage[frontend_id] = f"${backend_key}"
            else:
                 # Default projection
                 project_stage[frontend_id] = f"${backend_key}"

        # Add fields from Config
        for r in config.rows: add_project_field(r.id)
        for c in config.columns: add_project_field(c.id)
        for v in config.values: add_project_field(v.fieldId)
        for f in config.filters: add_project_field(f.id)
        
        pipeline.append({ "$project": project_stage })

        # 2. Match Stage (Filters)
        # TODO: Add specific value filters if passed in config (currently Config.filters is list of fields, 
        # but user might want to say "Status = Open"). 
        # Assuming config.filters is just "fields available to filter"? 
        # The PivotConfig interface Step 55 shows `filters: PivotField[]`. 
        # It doesn't seem to hold the *values* to filter by. 
        # So we skip actual row filtering for now unless we add a new property to request.
        
        # 3. Limit (for safety)
        if config.limit:
             pipeline.append({ "$limit": config.limit })
        # else:
        #      pipeline.append({ "$limit": 5000 }) # Safety cap removed per user request for "All Time"

        return pipeline

