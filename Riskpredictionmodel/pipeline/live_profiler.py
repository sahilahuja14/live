from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..config import get_live_db_name
from ..dbconnect import get_live_database
from ..logging_config import get_logger
from .live_field_map import LIVE_PROFILE_COLLECTIONS
from .utils import flatten_dict, is_missing, json_safe, safe_ratio, write_json


logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pipeline" / "schema_inventory"


def _normalize_join_value(value: Any) -> str | None:
    if is_missing(value):
        return None
    return str(value)


def _sample_documents(collection, sample_size: int) -> list[dict]:
    return list(collection.find({}).limit(int(sample_size)))


def _profile_documents(documents: Iterable[dict], *, sample_size: int) -> dict[str, Any]:
    docs = list(documents)
    field_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"non_missing": 0, "types": set(), "samples": []})
    total = len(docs)
    for raw_doc in docs:
        flat = flatten_dict(raw_doc)
        for path, value in flat.items():
            stats = field_stats[path]
            if not is_missing(value):
                stats["non_missing"] += 1
                stats["types"].add(type(value).__name__)
                if len(stats["samples"]) < 3:
                    stats["samples"].append(json_safe(value))
            elif len(stats["samples"]) < 3:
                stats["samples"].append(None)

    return {
        "sample_size": int(min(total, sample_size)),
        "fields": {
            path: {
                "types": sorted(stats["types"]),
                "non_missing_rate": round(safe_ratio(stats["non_missing"], total) * 100.0, 2),
                "samples": stats["samples"],
            }
            for path, stats in sorted(field_stats.items())
        },
    }


def profile_collection(collection_name: str, *, sample_size: int = 500, db=None) -> dict[str, Any]:
    database = db if db is not None else get_live_database()
    collection = database[collection_name]
    documents = _sample_documents(collection, sample_size=sample_size)
    profile = _profile_documents(documents, sample_size=sample_size)
    profile["collection"] = collection_name
    profile["database"] = database.name
    profile["document_count_profiled"] = len(documents)
    return profile


def find_join_keys(
    *,
    invoice_collection: str = "invoicemasters",
    payment_collection: str = "paymenttransactions",
    sample_size: int = 500,
    db=None,
) -> dict[str, Any]:
    database = db if db is not None else get_live_database()
    invoice_docs = _sample_documents(database[invoice_collection], sample_size=sample_size)
    payment_docs = _sample_documents(database[payment_collection], sample_size=sample_size)

    invoice_flats = [flatten_dict(doc) for doc in invoice_docs]
    payment_flats = [flatten_dict(doc) for doc in payment_docs]
    invoice_candidates = ("_id", "invoiceNo", "legacy.invoice_ref_raw")
    payment_candidates = ("performaInvoiceId", "finalInvoiceId", "invoiceId", "invoiceNo")

    comparisons: list[dict[str, Any]] = []
    for invoice_field in invoice_candidates:
        invoice_values = {
            normalized
            for flat in invoice_flats
            if (normalized := _normalize_join_value(flat.get(invoice_field))) is not None
        }
        for payment_field in payment_candidates:
            payment_values = [
                normalized
                for flat in payment_flats
                if (normalized := _normalize_join_value(flat.get(payment_field))) is not None
            ]
            match_count = sum(1 for value in payment_values if value in invoice_values)
            comparisons.append(
                {
                    "invoice_field": invoice_field,
                    "payment_field": payment_field,
                    "payment_rows": len(payment_values),
                    "match_pct": round(safe_ratio(match_count, len(payment_values)) * 100.0, 2),
                }
            )

    best = max(comparisons, key=lambda item: item["match_pct"]) if comparisons else None
    return {
        "invoice_collection": invoice_collection,
        "payment_collection": payment_collection,
        "best_match": best,
        "comparisons": comparisons,
    }


def profile_live_database(*, sample_size: int = 500, collection_names: list[str] | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    database = get_live_database()
    names = collection_names or sorted(database.list_collection_names())
    payload = {
        "database": get_live_db_name() or database.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collections": {name: profile_collection(name, sample_size=sample_size, db=database) for name in names},
        "join_keys": find_join_keys(sample_size=sample_size, db=database),
    }
    target_dir = output_dir or DEFAULT_OUTPUT_DIR
    write_json(target_dir / "live_schema_inventory.json", payload)
    logger.info("Live schema inventory written to %s", target_dir)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile livedb collections for Riskpredictionmodel cutover.")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--collections", nargs="*", default=list(LIVE_PROFILE_COLLECTIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    profile_live_database(
        sample_size=max(int(args.sample_size), 1),
        collection_names=list(args.collections) if args.collections else None,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
