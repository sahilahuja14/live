from fastapi import APIRouter, HTTPException, Depends
from ..schemas.analytics import PivotConfig
from ..services.query_builder import QueryBuilder
from ..database import get_all_async_mode_databases
import asyncio
from typing import List, Dict, Any

router = APIRouter(tags=["analytics"])

@router.post("/query")
async def query_analytics(config: PivotConfig):
    try:
        mode_dbs = get_all_async_mode_databases()
        
        qb = QueryBuilder()
        pipeline = qb.build_pipeline(config)
        
        async def run_query(db):
            cursor = await db.queries.aggregate(pipeline)
            return await cursor.to_list(length=None)
            
        tasks = [run_query(db) for db in mode_dbs.values()]
        all_results = await asyncio.gather(*tasks)
        
        results = []
        for res_list in all_results:
            results.extend(res_list)
        
        # Serialize _id if it's an ObjectId (though aggregation _id is usually our group key)
        # If _id is a dict (group keys), it's fine.
        
        # Flatten results for frontend table if needed?
        # Frontend PivotDisplay expects a list of objects.
        # The result of group is { _id: { row1: val, col1: val }, metric_sum: 100 }
        # We should flatten this for easier table display: 
        # { row1: val, col1: val, metric_sum: 100 }
        
        flattened_results = []
        for doc in results:
            flat = {}
            # Flatten _id
            if isinstance(doc.get('_id'), dict):
                for k, v in doc['_id'].items():
                    flat[k] = v
            elif doc.get('_id') is not None:
                flat['group'] = doc['_id'] # Single group key
                
            # Add metrics
            for k, v in doc.items():
                if k != '_id':
                    flat[k] = v
                    
            flattened_results.append(flat)
            
        return { "data": flattened_results }
        
    except Exception as e:
        print(f"Analytics Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
