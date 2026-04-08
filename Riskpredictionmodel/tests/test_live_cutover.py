from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from Riskpredictionmodel.features.production_registry import validate_feature_frame
from Riskpredictionmodel.pipeline.live_adapter import _normalize_live_doc, compute_live_coverage, fetch_live_frame, join_payment_transactions
from Riskpredictionmodel.pipeline.live_field_map import LIVE_PASSTHROUGH_FIELDS, LIVE_TO_NORMALIZED
from Riskpredictionmodel.pipeline.risk_canonical import NORMALIZED_TO_CANONICAL_FIELD_MAP, RISK_MAIN_FETCH_PROJECTION, canonicalize_risk_main_frame
from Riskpredictionmodel.pipeline.risk_main import build_risk_main_scoring_frame
from Riskpredictionmodel.scoring.model import load_production_artifacts


class _FakeCursor:
    def __init__(self, rows):
        self._rows = list(rows)

    def batch_size(self, _value):
        return self

    def limit(self, value):
        return _FakeCursor(self._rows[: int(value)])

    def __iter__(self):
        return iter(self._rows)


class _FakeCollection:
    def __init__(self, rows):
        self.rows = list(rows)

    def find(self, query=None, projection=None, limit=None):
        query = query or {}
        if "finalInvoiceId" in query:
            values = {str(item) for item in query["finalInvoiceId"].get("$in", [])}
            filtered = [row for row in self.rows if str(row.get("finalInvoiceId")) in values]
            return _FakeCursor(filtered)
        return _FakeCursor(self.rows if limit is None else self.rows[: int(limit)])


class _FakeDatabase:
    def __init__(self, collections):
        self.collections = collections
        self.name = "livedb"

    def __getitem__(self, key):
        return self.collections[key]

    def list_collection_names(self):
        return sorted(self.collections)


class LiveCutoverTests(unittest.TestCase):
    def _live_invoice_docs(self):
        return [
            {
                "_id": "mongo-001",
                "invoiceNo": "INV-001",
                "invoiceDate": "2025-10-01",
                "dueDate": "2025-10-31",
                "paymentTerms": 30,
                "currency": "USD",
                "taxableAmount": 100000,
                "totalAmount": 112000,
                "status": "Paid",
                "companyCode": "ZIPA",
                "salesOwner": "Riya Singh",
                "customer": {
                    "id": "CUST-001",
                    "name": "Acme Logistics",
                    "accountType": "Credit",
                    "type": "Existing",
                    "category": "Key",
                    "onboardDate": "2024-01-15",
                },
                "shipment": {
                    "queryFor": "air",
                    "accountType": "Export",
                    "incoTerms": "FOB",
                    "commodity": "Electronics",
                    "grossWeight": 1200,
                    "chargeableWeight": 1250,
                    "volumeWeight": 1180,
                    "noOfContainers": 0,
                    "originCity": "Delhi",
                    "originState": "Delhi",
                    "originCountry": "India",
                    "destinationCity": "Dubai",
                    "destinationState": "Dubai",
                    "destinationCountry": "UAE",
                },
                "receivables": {
                    "notDue": 10000,
                    "0to15": 5000,
                    "16to30": 7000,
                    "31to45": 0,
                    "46to60": 0,
                    "60to90": 0,
                    "above90": 0,
                },
                "executionDate": "2025-09-28",
                "paidAmount": 100000,
                "tdsAmount": 2000,
                "ytdExposure": 210000,
                "operational": {
                    "jobNo": "JOB-1",
                    "bookingNo": "BOOK-1",
                    "clearanceStatus": "Cleared",
                    "gateInStatus": "Received",
                    "lastTrackingStatus": "Delivered",
                    "lastTrackingLocation": "Dubai",
                    "shippingBillNo": "SB-1",
                    "containerNo": "CONT-1",
                },
            },
            {
                "_id": "mongo-002",
                "invoiceNo": "INV-002",
                "invoiceDate": "2025-11-12",
                "dueDate": "2025-12-12",
                "paymentTerms": 30,
                "currency": "USD",
                "taxableAmount": 160000,
                "totalAmount": 179000,
                "status": "Paid",
                "companyCode": "ZIPA",
                "salesOwner": "Riya Singh",
                "customer": {
                    "id": "CUST-001",
                    "name": "Acme Logistics",
                    "accountType": "Credit",
                    "type": "Existing",
                    "category": "Key",
                    "onboardDate": "2024-01-15",
                },
                "shipment": {
                    "queryFor": "air",
                    "accountType": "Export",
                    "incoTerms": "FOB",
                    "commodity": "Electronics",
                    "grossWeight": 1400,
                    "chargeableWeight": 1480,
                    "volumeWeight": 1430,
                    "noOfContainers": 0,
                    "originCity": "Delhi",
                    "originState": "Delhi",
                    "originCountry": "India",
                    "destinationCity": "Dubai",
                    "destinationState": "Dubai",
                    "destinationCountry": "UAE",
                },
                "receivables": {
                    "notDue": 12000,
                    "0to15": 7000,
                    "16to30": 10000,
                    "31to45": 12000,
                    "46to60": 6000,
                    "60to90": 3000,
                    "above90": 0,
                },
                "executionDate": "2025-11-08",
                "paidAmount": 120000,
                "tdsAmount": 3000,
                "ytdExposure": 250000,
                "operational": {
                    "jobNo": "JOB-2",
                    "bookingNo": "BOOK-2",
                    "clearanceStatus": "Hold",
                    "gateInStatus": "Received",
                    "lastTrackingStatus": "In Transit",
                    "lastTrackingLocation": "Mumbai",
                    "shippingBillNo": "SB-2",
                    "containerNo": "CONT-2",
                },
            },
        ]

    def _payment_docs(self):
        return [
            {"finalInvoiceId": "INV-001", "paymentDate": "2025-10-25", "status": "settled", "paymentDetails": "first"},
            {"finalInvoiceId": "INV-001", "paymentDate": "2025-11-05", "status": "settled", "paymentDetails": "second"},
            {"finalInvoiceId": "INV-002", "paymentDate": "20-01-2026", "status": "paid", "paymentDetails": "single"},
        ]

    def test_live_mapping_contract_matches_risk_map(self):
        allowed = set(NORMALIZED_TO_CANONICAL_FIELD_MAP)
        mapped_values = {value for value in LIVE_TO_NORMALIZED.values() if value is not None}
        self.assertTrue(mapped_values.issubset(allowed))

        required_projection = {
            key
            for key in RISK_MAIN_FETCH_PROJECTION
            if key != "_id" and key not in {"paymentDate_raw", "paymentDetailsRaw", "legacy.invoice_ref_raw", "companyCode"}
        }
        self.assertTrue(required_projection.issubset(mapped_values))
        self.assertEqual(
            set(LIVE_PASSTHROUGH_FIELDS.values()),
            {"companyCode", "paymentDate_raw", "paymentDetailsRaw", "legacy.invoice_ref_raw"},
        )

    def test_normalize_live_doc_flattens_and_maps(self):
        doc = self._live_invoice_docs()[0]
        normalized = _normalize_live_doc(doc)
        self.assertEqual(normalized["invoiceNo"], "INV-001")
        self.assertEqual(normalized["customer.customerId"], "CUST-001")
        self.assertEqual(normalized["shipmentDetails.queryFor"], "air")
        self.assertEqual(normalized["companyCode"], "ZIPA")
        self.assertNotIn("unknownField", normalized)

    def test_join_payment_transactions_uses_latest_settled_date(self):
        frame = pd.DataFrame([_normalize_live_doc(self._live_invoice_docs()[0])])
        joined = join_payment_transactions(frame, payment_rows=self._payment_docs())
        self.assertEqual(str(joined.loc[0, "paymentDate"])[:10], "2025-11-05")
        self.assertEqual(int(joined.loc[0, "paymentInstallmentCount"]), 2)
        self.assertTrue(bool(joined.loc[0, "partialPaymentFlag"]))

    def test_fetch_live_frame_reports_coverage(self):
        fake_db = _FakeDatabase(
            {
                "invoicemasters": _FakeCollection(self._live_invoice_docs()),
                "paymenttransactions": _FakeCollection(self._payment_docs()),
            }
        )
        with patch.dict(
            os.environ,
            {"PRODUCTION_RISK_SOURCE_MODE": "live_collections", "LIVE_COLLECTION_INVOICES": "invoicemasters"},
            clear=False,
        ), patch(
            "Riskpredictionmodel.pipeline.live_adapter.get_live_database",
            return_value=fake_db,
        ):
            frame = fetch_live_frame(limit=10)
            coverage = compute_live_coverage(frame)
            self.assertEqual(len(frame), 2)
            self.assertGreaterEqual(coverage["invoice_key_pct"], 100.0)
            self.assertGreaterEqual(coverage["customer_id_pct"], 100.0)

    def test_live_frame_builds_complete_feature_contract(self):
        fake_db = _FakeDatabase(
            {
                "invoicemasters": _FakeCollection(self._live_invoice_docs()),
                "paymenttransactions": _FakeCollection(self._payment_docs()),
            }
        )
        with patch.dict(
            os.environ,
            {"PRODUCTION_RISK_SOURCE_MODE": "live_collections", "LIVE_COLLECTION_INVOICES": "invoicemasters"},
            clear=False,
        ), patch(
            "Riskpredictionmodel.pipeline.live_adapter.get_live_database",
            return_value=fake_db,
        ):
            live_frame = fetch_live_frame(limit=10)
        canonical = canonicalize_risk_main_frame(live_frame)
        scoring_frame = build_risk_main_scoring_frame(live_frame.iloc[:1].copy(), history_df=live_frame)
        artifacts = load_production_artifacts(force_reload=True)
        validation = validate_feature_frame(scoring_frame, artifacts["features"])

        self.assertFalse(canonical.empty)
        self.assertEqual(len(artifacts["features"]), 323)
        self.assertTrue(validation.is_valid, msg=str(validation.domain_gaps))


if __name__ == "__main__":
    unittest.main()
