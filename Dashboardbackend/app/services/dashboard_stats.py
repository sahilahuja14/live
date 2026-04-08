from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict
from copy import deepcopy
from bson import ObjectId

from ..database import get_async_analytics_database, get_all_async_queryFor_databases
import asyncio


# -------------------------------------------------
# CONSTANT TEMPLATES
# -------------------------------------------------

EMPTY_STATS = {
    "queries": {"open": 0, "rates_available": 0, "rates_quoted": 0, "rates_confirmed": 0, "lost": 0},
    "bookings": {"booking": 0, "pending": 0, "pricing_approval": 0},
    "shipment": {"created": 0, "final": 0, "executed": 0},
    "invoice": {"total": 0},
    "weight": {"charge": 0, "gross": 0}
}

EMPTY_FINANCE = {
    "Turnover": 0.0,
    "Margin": 0.0,
    "Tonnage": 0.0,
    "Buy": 0.0,
    "Count": 0
}


def get_empty_stats():
    return deepcopy(EMPTY_STATS)


def get_empty_finance_stats():
    return deepcopy(EMPTY_FINANCE)


# -------------------------------------------------
# STATUS MAP
# -------------------------------------------------

STATUS_MAP = {
    "Open": ("queries", "open"),
    "Rates Available": ("queries", "rates_available"),
    "Rates Quoted": ("queries", "rates_quoted"),
    "Rates Confirmed": ("queries", "rates_confirmed"),
    "Shipment Lost": ("queries", "lost"),

    "Booking": ("bookings", "booking"),
    "Pending": ("bookings", "pending"),
    "Pricing Approval": ("bookings", "pricing_approval"),

    "Shipment Created": ("shipment", "created"),
    "Shipment Final": ("shipment", "final"),
    "Shipment Executed": ("shipment", "executed")
}

QUERYFOR_MAP = {
    "air": "Air",
    "ocean": "Ocean",
    "road": "Road",
    "courier": "Courier"
}


# -------------------------------------------------
# HELPER
# -------------------------------------------------

def update_stats(stats, doc):
    status = doc.get("quoteStatus")

    if status in STATUS_MAP:
        group, key = STATUS_MAP[status]
        stats[group][key] += 1

    if status in ("Invoice", "Paid"):
        stats["invoice"]["total"] += 1

    cw = float(doc.get("chargeableWeight") or 0)
    gw = float(doc.get("grossWeight") or 0)

    stats["weight"]["charge"] += cw / 1000.0
    stats["weight"]["gross"] += int(gw)


# -------------------------------------------------
# DASHBOARD STATS
# -------------------------------------------------

async def calculate_dashboard_stats(
    range_param: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    shipment_type: str = "all",
    query_type: Optional[str] = None
):
    queryFor_dbs = get_all_async_queryFor_databases()
    # No single queries_col anymore

    end_date = to_date or datetime.now()
    start_date = from_date

    match_query = {}

    if range_param != 'all':
        if not start_date:
            ranges = {
                "daily": 1,
                "weekly": 7,
                "monthly": 30,
                "quarterly": 90,
                "yearly": 365
            }
            start_date = end_date - timedelta(days=ranges.get(range_param, 365))

        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

        match_query["createdAt"] = {"$gte": start_date, "$lte": end_date}

    if shipment_type and shipment_type.lower() != "all":
        queryFors = [QUERYFOR_MAP.get(m.strip().lower()) for m in shipment_type.split(",")]
        match_query["queryFor"] = {"$in": queryFors}

    if query_type and query_type.lower() != "all":
        match_query["queryType"] = {"$regex": query_type, "$options": "i"}

    # -------------------------------------------------
    # PIPELINE CONSTANTS
    # -------------------------------------------------
    def build_agg_sums():
        group_fields = {}
        for group_name in ["queries", "bookings", "shipment"]:
            keys = set(k for g, k in STATUS_MAP.values() if g == group_name)
            for key in keys:
                matching_statuses = [status for status, (g, k) in STATUS_MAP.items() if g == group_name and k == key]
                group_fields[f"{group_name}_{key}"] = {"$sum": {"$cond": [{"$in": ["$quoteStatus", matching_statuses]}, 1, 0]}}
                
        group_fields["invoice_total"] = {"$sum": {"$cond": [{"$in": ["$quoteStatus", ["Shipment Executed", "Shipment Created"]]}, 1, 0]}}
        group_fields["weight_charge"] = {"$sum": {"$divide": [{"$convert": {"input": "$chargeableWeight", "to": "double", "onError": 0.0, "onNull": 0.0}}, 1000.0]}}
        group_fields["weight_gross"] = {"$sum": {"$convert": {"input": "$grossWeight", "to": "int", "onError": 0, "onNull": 0}}}
        group_fields["count"] = {"$sum": 1}
        return group_fields

    def unpack_agg_stats(data_dict):
        stats = get_empty_stats()
        stats["queries"]["open"] = data_dict.get("queries_open", 0)
        stats["queries"]["rates_available"] = data_dict.get("queries_rates_available", 0)
        stats["queries"]["rates_quoted"] = data_dict.get("queries_rates_quoted", 0)
        stats["queries"]["rates_confirmed"] = data_dict.get("queries_rates_confirmed", 0)
        stats["queries"]["lost"] = data_dict.get("queries_lost", 0)
        
        stats["bookings"]["booking"] = data_dict.get("bookings_booking", 0)
        stats["bookings"]["pending"] = data_dict.get("bookings_pending", 0)
        stats["bookings"]["pricing_approval"] = data_dict.get("bookings_pricing_approval", 0)
        
        stats["shipment"]["created"] = data_dict.get("shipment_created", 0)
        stats["shipment"]["final"] = data_dict.get("shipment_final", 0)
        stats["shipment"]["executed"] = data_dict.get("shipment_executed", 0)
        
        stats["invoice"]["total"] = data_dict.get("invoice_total", 0)
        stats["weight"]["charge"] = data_dict.get("weight_charge", 0)
        stats["weight"]["gross"] = data_dict.get("weight_gross", 0)
        return stats

    pipeline = [
        {"$match": match_query},
        {"$addFields": {
            "normqueryFor": {
                "$switch": {
                    "branches": [
                        {"case": {"$regexMatch": {"input": {"$ifNull": ["$queryFor", ""]}, "regex": "air", "options": "i"}}, "then": "Air"},
                        {"case": {"$regexMatch": {"input": {"$ifNull": ["$queryFor", ""]}, "regex": "ocean", "options": "i"}}, "then": "Ocean"},
                        {"case": {"$regexMatch": {"input": {"$ifNull": ["$queryFor", ""]}, "regex": "road", "options": "i"}}, "then": "Road"},
                        {"case": {"$regexMatch": {"input": {"$ifNull": ["$queryFor", ""]}, "regex": "courier", "options": "i"}}, "then": "Courier"}
                    ],
                    "default": "Other"
                }
            },
            "normQueryType": {
                "$switch": {
                    "branches": [
                        {"case": {"$regexMatch": {"input": {"$ifNull": ["$queryType", ""]}, "regex": "import", "options": "i"}}, "then": "Import"},
                        {"case": {"$regexMatch": {"input": {"$ifNull": ["$queryType", ""]}, "regex": "domestic", "options": "i"}}, "then": "Domestic"},
                        {"case": {"$regexMatch": {"input": {"$ifNull": ["$queryType", ""]}, "regex": "third", "options": "i"}}, "then": "Third Country"}
                    ],
                    "default": "Export"
                }
            },
            "dateStr": {
                "$cond": [
                    {"$eq": [{"$type": "$createdAt"}, "date"]},
                    {"$dateToString": {"format": "%Y-%m-%d", "date": "$createdAt"}},
                    {"$literal": "Unknown"}
                ]
            },
            "routeStr": {
                "$concat": [
                    {"$ifNull": ["$originAirport.code", "Unknown"]}, 
                    " -> ", 
                    {"$ifNull": ["$destinationAirport.code", "Unknown"]}
                ]
            }
        }},
        {"$facet": {
            "global_totals": [
                {"$group": {"_id": None, **build_agg_sums()}}
            ],
            "daily_grouped": [
                {"$group": {
                    "_id": {
                        "date": "$dateStr",
                        "queryFor": "$normqueryFor",
                        "qType": "$normQueryType"
                    },
                    **build_agg_sums()
                }}
            ],
            "clients": [
                {"$group": {"_id": {"$ifNull": ["$customerName", "Unknown"]}, "count": {"$sum": 1}}}
            ],
            "routes": [
                {"$group": {"_id": "$routeStr", "count": {"$sum": 1}}}
            ]
        }}
    ]

    async def run_agg(db):
        cursor = await db["queries"].aggregate(pipeline)
        result = await cursor.to_list(length=1)
        return result[0] if result and result[0] else {}

    tasks = [run_agg(db) for db in queryFor_dbs.values()]
    queryFor_results = await asyncio.gather(*tasks)

    # Merge results from all queryFors
    merged_global = {}
    merged_daily = []
    merged_clients = []
    merged_routes = []

    for res in queryFor_results:
        # Sum global totals
        for gt in res.get("global_totals", []):
            for k, v in gt.items():
                if k == "_id": continue
                merged_global[k] = merged_global.get(k, 0) + (v or 0)
        
        merged_daily.extend(res.get("daily_grouped", []))
        merged_clients.extend(res.get("clients", []))
        merged_routes.extend(res.get("routes", []))
        
    global_totals = unpack_agg_stats(merged_global)

    client_counts = defaultdict(int)
    for item in merged_clients:
        if item["_id"] != "Unknown":
            client_counts[item["_id"]] += item["count"]

    route_counts = defaultdict(int)
    for item in merged_routes:
        route_counts[item["_id"]] += item["count"]

    history_map = {}
    by_queryFor_totals = {m: get_empty_stats() for m in ['All', 'Air', 'Ocean', 'Road', 'Courier']}
    by_query_type_totals = {q: get_empty_stats() for q in ['All', 'Import', 'Export', 'Domestic', 'Third Country']}

    def add_stats(target, source):
        # Merge two stat dictionaries
        for rk, rv in source.items():
            for kk, vv in rv.items():
                target[rk][kk] += vv

    for group in merged_daily:
        date_str = group["_id"]["date"]
        if date_str == "Unknown":
            continue
            
        queryFor = group["_id"]["queryFor"]
        q_type = group["_id"]["qType"]
        stats = unpack_agg_stats(group)
        
        if date_str not in history_map:
            history_map[date_str] = {
                'All': get_empty_stats(),
                'Air': get_empty_stats(),
                'Ocean': get_empty_stats(),
                'Road': get_empty_stats(),
                'Courier': get_empty_stats(),
                'byQueryType': {
                    'Import': get_empty_stats(),
                    'Export': get_empty_stats(),
                    'Domestic': get_empty_stats(),
                    'Third Country': get_empty_stats()
                }
            }
            
        add_stats(history_map[date_str]['All'], stats)
        if queryFor in history_map[date_str]:
            add_stats(history_map[date_str][queryFor], stats)
        if q_type in history_map[date_str]['byQueryType']:
            add_stats(history_map[date_str]['byQueryType'][q_type], stats)
            
        add_stats(by_queryFor_totals['All'], stats)
        if queryFor in by_queryFor_totals:
            add_stats(by_queryFor_totals[queryFor], stats)
            
        add_stats(by_query_type_totals['All'], stats)
        if q_type in by_query_type_totals:
            add_stats(by_query_type_totals[q_type], stats)

    processed_history = []
    for d in sorted(history_map.keys()):
        day_data = history_map[d]
        processed_history.append({
            "date": d,
            **day_data["All"],
            "byqueryFor": {m: day_data[m] for m in ['Air', 'Ocean', 'Road', 'Courier']},
            "byQueryType": day_data["byQueryType"]
        })

    return {
        **global_totals,
        "history": processed_history,
        "byqueryFor": by_queryFor_totals,
        "byQueryType": by_query_type_totals,
        "byClient": dict(client_counts),
        "byRoute": dict(route_counts)
    }


# -------------------------------------------------
# FINANCIAL STATS
# -------------------------------------------------

async def calculate_financial_stats(
    range_param: Optional[str] = "monthly",
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    modes: Optional[list] = None
):

    db = get_async_analytics_database()

    invoices_col = db["invoicemasters"]
    buysell_col = db["financebuysellmasters"]

    end_date = to_date or datetime.now()
    start_date = from_date

    match_query = {"isDeleted": False}

    if range_param != "all":

        ranges = {
            "daily": 1,
            "weekly": 7,
            "monthly": 30,
            "quarterly": 90,
            "yearly": 365
        }

        if not start_date:
            start_date = end_date - timedelta(days=ranges.get(range_param, 30))

        match_query["invoiceDate"] = {
            "$gte": start_date,
            "$lte": end_date
        }

    if modes:
        match_query["shipmentDetails.queryFor"] = {"$in": modes}

    pipeline = [
        {"$match": match_query},
        {
            "$lookup": {
                "from": "financebuysellmasters",
                "let": {
                    "buyId": {
                        "$convert": {
                            "input": "$buySaleId",
                            "to": "objectId",
                            "onError": None,
                            "onNull": None
                        }
                    }
                },
                "pipeline": [
                    {"$match": {"$expr": {"$eq": ["$_id", "$$buyId"]}}},
                    {"$project": {"_id": 0, "totalBuy": "$buyValue.totalChargeTaxableB"}}
                ],
                "as": "buy_info"
            }
        },
        {
            "$addFields": {
                "rawTurnover": {"$convert": {"input": "$taxableTotalAmountB", "to": "double", "onError": 0.0, "onNull": 0.0}},
                "rawBuy": {
                    "$convert": {
                        "input": {"$ifNull": [{"$arrayElemAt": ["$buy_info.totalBuy", 0]}, 0.0]},
                        "to": "double",
                        "onError": 0.0,
                        "onNull": 0.0
                    }
                },
                "isCN": {"$eq": ["$invoiceType", "CN"]},
                "tonnage": {
                    "$divide": [
                        {"$convert": {"input": {"$ifNull": ["$shipmentDetails.chargeableWeight", 0]}, "to": "double", "onError": 0.0, "onNull": 0.0}},
                        1000.0
                    ]
                },
                "dateStr": {
                    "$cond": [
                        {"$eq": [{"$type": "$invoiceDate"}, "date"]},
                        {"$dateToString": {"format": "%Y-%m-%d", "date": "$invoiceDate"}},
                        "Unknown"
                    ]
                },
                "mode": {"$ifNull": ["$shipmentDetails.queryFor", "UNKNOWN"]},
                "customerName": {"$ifNull": ["$customer.customerName", "UNKNOWN"]}
            }
        },
        {
            "$addFields": {
                "turnover": {
                    "$cond": [
                        "$isCN",
                        {"$multiply": ["$rawTurnover", -1]},
                        "$rawTurnover"
                    ]
                },
                "buy": {
                    "$cond": [
                        "$isCN",
                        {"$multiply": ["$rawBuy", -1]},
                        "$rawBuy"
                    ]
                }
            },
        },
        {
            "$addFields": {
                "margin": {"$subtract": ["$turnover", "$buy"]},
                "routeStr": {
                    "$cond": [
                        {"$eq": ["$mode", "Air"]},
                        {"$concat": [{"$ifNull": ["$shipmentDetails.originAirportName", "UNKNOWN"]}, " → ", {"$ifNull": ["$shipmentDetails.destinationAirportName", "UNKNOWN"]}]},
                        {"$concat": [{"$ifNull": ["$shipmentDetails.originName", "UNKNOWN"]}, " → ", {"$ifNull": ["$shipmentDetails.destinationName", "UNKNOWN"]}]}
                    ]
                },
                "carrierStr": {
                    "$cond": [
                        {"$eq": ["$mode", "Air"]},
                        {"$ifNull": ["$shipmentDetails.airlineName", "UNKNOWN"]},
                        {"$cond": [
                            {"$eq": ["$mode", "Ocean"]},
                            {"$ifNull": ["$shipmentDetails.shippingLineName", "UNKNOWN"]},
                            "N/A"
                        ]}
                    ]
                }
            }
        },
        {
            "$facet": {
                "global_totals": [
                    {
                        "$group": {
                            "_id": None,
                            "Turnover": {"$sum": "$turnover"},
                            "Margin": {"$sum": "$margin"},
                            "Tonnage": {"$sum": "$tonnage"},
                            "Buy": {"$sum": "$buy"},
                            "Count": {"$sum": 1}
                        }
                    }
                ],
                "risk_metrics": [
                    {"$match": {"turnover": {"$ne": 0.0}}},
                    {
                        "$project": {
                            "margin": 1,
                            "marginPct": {"$multiply": [{"$divide": ["$margin", "$turnover"]}, 100]}
                        }
                    },
                    {
                        "$group": {
                            "_id": None,
                            "loss_making": {"$sum": {"$cond": [{"$lt": ["$margin", 0]}, 1, 0]}},
                            "low_margin": {"$sum": {"$cond": [{"$and": [{"$gt": ["$marginPct", 0]}, {"$lt": ["$marginPct", 5]}]}, 1, 0]}}
                        }
                    }
                ],
                "daily": [
                    {"$group": {"_id": "$dateStr", "Turnover": {"$sum": "$turnover"}, "Margin": {"$sum": "$margin"}, "Tonnage": {"$sum": "$tonnage"}, "Buy": {"$sum": "$buy"}, "Count": {"$sum": 1}}}
                ],
                "byCustomer": [
                    {"$group": {"_id": {"$ifNull": ["$customerName", "UNKNOWN"]}, "Turnover": {"$sum": "$turnover"}, "Margin": {"$sum": "$margin"}, "Tonnage": {"$sum": "$tonnage"}, "Buy": {"$sum": "$buy"}, "Count": {"$sum": 1}}}
                ],
                "byRoute": [
                    {"$group": {"_id": {"$ifNull": ["$routeStr", "UNKNOWN"]}, "Turnover": {"$sum": "$turnover"}, "Margin": {"$sum": "$margin"}, "Tonnage": {"$sum": "$tonnage"}, "Buy": {"$sum": "$buy"}, "Count": {"$sum": 1}}}
                ],
                "byCarrier": [
                    {"$group": {"_id": {"$ifNull": ["$carrierStr", "UNKNOWN"]}, "Turnover": {"$sum": "$turnover"}, "Margin": {"$sum": "$margin"}, "Tonnage": {"$sum": "$tonnage"}, "Buy": {"$sum": "$buy"}, "Count": {"$sum": 1}}}
                ],
                "granular": [
                    {
                        "$group": {
                            "_id": {
                                "date": "$dateStr",
                                "mode": "$mode",
                                "customer": "$customerName",
                                "route": "$routeStr",
                                "carrier": "$carrierStr"
                            },
                            "Turnover": {"$sum": "$turnover"},
                            "Margin": {"$sum": "$margin"},
                            "Tonnage": {"$sum": "$tonnage"},
                            "Buy": {"$sum": "$buy"},
                            "Count": {"$sum": 1}
                        }
                    }
                ]
            }
        }
    ]

    cursor = await invoices_col.aggregate(pipeline)
    res = await cursor.to_list(length=1)

    if not res or not res[0] or not res[0].get("global_totals"):
        return {
            "summary": get_empty_finance_stats(),
            "history": [],
            "byCustomer": {},
            "byRoute": {},
            "byCarrier": {},
            "granular": [],
            "risk": {"low_margin": 0, "loss_making": 0}
        }

    agg_res = res[0]

    def map_finance_group(group_item):
        return {
            "Turnover": group_item.get("Turnover", 0.0),
            "Margin": group_item.get("Margin", 0.0),
            "Tonnage": group_item.get("Tonnage", 0.0),
            "Buy": group_item.get("Buy", 0.0),
            "Count": group_item.get("Count", 0)
        }

    summary = map_finance_group(agg_res["global_totals"][0])
    
    risk_data = agg_res.get("risk_metrics", [{}])[0] if agg_res.get("risk_metrics") else {}
    risk = {
        "low_margin": risk_data.get("low_margin", 0),
        "loss_making": risk_data.get("loss_making", 0)
    }

    history_list = [{"date": item["_id"], **map_finance_group(item)} for item in agg_res.get("daily", []) if item["_id"] != "Unknown"]
    history_list.sort(key=lambda x: x["date"])

    customer_stats = {item["_id"]: map_finance_group(item) for item in agg_res.get("byCustomer", []) if item["_id"] != "UNKNOWN"}
    route_stats = {item["_id"]: map_finance_group(item) for item in agg_res.get("byRoute", []) if item["_id"] not in ("UNKNOWN", "UNKNOWN → UNKNOWN", " → ", "None → None")}
    carrier_stats = {item["_id"]: map_finance_group(item) for item in agg_res.get("byCarrier", []) if item["_id"] not in ("UNKNOWN", "N/A", "None")}

    granular_list = []
    for item in agg_res.get("granular", []):
        if item["_id"]["date"] == "Unknown":
            continue
        granular_list.append({
            "date": item["_id"]["date"],
            "mode": item["_id"]["mode"],
            "customer": item["_id"]["customer"],
            "route": item["_id"]["route"],
            "carrier": item["_id"]["carrier"],
            **map_finance_group(item)
        })

    return {
        "summary": summary,
        "history": history_list,
        "byCustomer": dict(sorted(customer_stats.items(), key=lambda x: x[1]['Count'], reverse=True)),
        "byRoute": dict(sorted(route_stats.items(), key=lambda x: x[1]['Count'], reverse=True)),
        "byCarrier": dict(sorted(carrier_stats.items(), key=lambda x: x[1]['Count'], reverse=True)),
        "granular": granular_list,
        "risk": risk
    }
