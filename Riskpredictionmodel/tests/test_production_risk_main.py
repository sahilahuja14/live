from __future__ import annotations

from datetime import datetime, timedelta
import os
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import jwt
import pandas as pd
from fastapi.testclient import TestClient
from pymongo.errors import ServerSelectionTimeoutError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from Riskpredictionmodel.features.customer_aggregates import build_customer_history_aggregates
from Riskpredictionmodel.pipeline.risk_main import build_risk_main_manual_request_frame, build_risk_main_scoring_frame
from Riskpredictionmodel.scoring.model import describe_active_production_model, load_production_artifacts
from Riskpredictionmodel.api import scoring_api
from Riskpredictionmodel.api.response_builder import response_from_raw


def _history_frame() -> pd.DataFrame:
    rows = [
        {
            "_id": "inv-001",
            "invoiceNo": "INV-001",
            "invoiceDate": "2025-10-01",
            "invoiceDueDate": "2025-10-31",
            "paymentDate": "2025-11-05",
            "paymentDate_raw": "05-11-2025",
            "paidStatus": "Paid",
            "paymentTerms": 30,
            "customer.customerId": "CUST-001",
            "customer.customerName": "Acme Logistics",
            "customer.customerAccountType": "Credit",
            "customer.customerType": "Existing",
            "customer.category": "Key",
            "customer.onboardDate": "2024-01-15",
            "customer.custCurrency": "USD",
            "salesPersonName": "Riya Singh",
            "invoiceType": "Invoice",
            "selectedCustomerCurrency": "USD",
            "taxableTotalAmountB": 100000,
            "totalAmountB": 112000,
            "paidAmount": 100000,
            "tdsAmount": 2000,
            "ytdExposure": 210000,
            "receivables.notDueAmount": 10000,
            "receivables.bucket0To15Amount": 5000,
            "receivables.bucket16To30Amount": 7000,
            "receivables.bucket31To45Amount": 0,
            "receivables.bucket46To60Amount": 0,
            "receivables.bucket60To90Amount": 0,
            "receivables.bucketAbove90Amount": 0,
            "shipmentDetails.queryFor": "air",
            "shipmentDetails.accountType": "Export",
            "shipmentDetails.incoTerms": "FOB",
            "shipmentDetails.commodity": "Electronics",
            "shipmentDetails.grossWeight": 1200,
            "shipmentDetails.chargeableWeight": 1250,
            "shipmentDetails.volumeWeight": 1180,
            "shipmentDetails.noOfContainers": 0,
            "shipmentDetails.originCity": "Delhi",
            "shipmentDetails.originState": "Delhi",
            "shipmentDetails.originCountry": "India",
            "shipmentDetails.destinationCity": "Dubai",
            "shipmentDetails.destinationState": "Dubai",
            "shipmentDetails.destinationCountry": "UAE",
            "executionDate": "2025-09-28",
            "operational.clearanceStatus": "Cleared",
            "operational.gateInStatus": "Received",
            "operational.lastTrackingStatus": "Delivered",
            "operational.lastTrackingLocation": "Dubai",
            "companyCode": "ZIPA",
        },
        {
            "_id": "inv-002",
            "invoiceNo": "INV-002",
            "invoiceDate": "2025-11-12",
            "invoiceDueDate": "2025-12-12",
            "paymentDate": "2026-01-20",
            "paymentDate_raw": "20-01-2026",
            "paidStatus": "Paid",
            "paymentTerms": 30,
            "customer.customerId": "CUST-001",
            "customer.customerName": "Acme Logistics",
            "customer.customerAccountType": "Credit",
            "customer.customerType": "Existing",
            "customer.category": "Key",
            "customer.onboardDate": "2024-01-15",
            "customer.custCurrency": "USD",
            "salesPersonName": "Riya Singh",
            "invoiceType": "Invoice",
            "selectedCustomerCurrency": "USD",
            "taxableTotalAmountB": 160000,
            "totalAmountB": 179000,
            "paidAmount": 120000,
            "tdsAmount": 3000,
            "ytdExposure": 250000,
            "receivables.notDueAmount": 12000,
            "receivables.bucket0To15Amount": 7000,
            "receivables.bucket16To30Amount": 10000,
            "receivables.bucket31To45Amount": 12000,
            "receivables.bucket46To60Amount": 6000,
            "receivables.bucket60To90Amount": 3000,
            "receivables.bucketAbove90Amount": 0,
            "shipmentDetails.queryFor": "air",
            "shipmentDetails.accountType": "Export",
            "shipmentDetails.incoTerms": "FOB",
            "shipmentDetails.commodity": "Electronics",
            "shipmentDetails.grossWeight": 1400,
            "shipmentDetails.chargeableWeight": 1480,
            "shipmentDetails.volumeWeight": 1430,
            "shipmentDetails.noOfContainers": 0,
            "shipmentDetails.originCity": "Delhi",
            "shipmentDetails.originState": "Delhi",
            "shipmentDetails.originCountry": "India",
            "shipmentDetails.destinationCity": "Dubai",
            "shipmentDetails.destinationState": "Dubai",
            "shipmentDetails.destinationCountry": "UAE",
            "executionDate": "2025-11-08",
            "operational.clearanceStatus": "Hold",
            "operational.gateInStatus": "Received",
            "operational.lastTrackingStatus": "In Transit",
            "operational.lastTrackingLocation": "Mumbai",
            "companyCode": "ZIPA",
        },
        {
            "_id": "inv-003",
            "invoiceNo": "INV-003",
            "invoiceDate": "2026-01-08",
            "invoiceDueDate": "2026-02-07",
            "paymentDate": pd.NaT,
            "paymentDate_raw": "",
            "paidStatus": "Pending",
            "paymentTerms": 30,
            "customer.customerId": "CUST-002",
            "customer.customerName": "Beacon Traders",
            "customer.customerAccountType": "Cash",
            "customer.customerType": "New",
            "customer.category": "Standard",
            "customer.onboardDate": "2025-10-01",
            "customer.custCurrency": "INR",
            "salesPersonName": "Aman Kapoor",
            "invoiceType": "Invoice",
            "selectedCustomerCurrency": "INR",
            "taxableTotalAmountB": 90000,
            "totalAmountB": 95000,
            "paidAmount": 0,
            "tdsAmount": 1000,
            "ytdExposure": 120000,
            "receivables.notDueAmount": 18000,
            "receivables.bucket0To15Amount": 5000,
            "receivables.bucket16To30Amount": 0,
            "receivables.bucket31To45Amount": 0,
            "receivables.bucket46To60Amount": 0,
            "receivables.bucket60To90Amount": 0,
            "receivables.bucketAbove90Amount": 0,
            "shipmentDetails.queryFor": "ocean",
            "shipmentDetails.accountType": "Import",
            "shipmentDetails.incoTerms": "CIF",
            "shipmentDetails.commodity": "Chemicals",
            "shipmentDetails.grossWeight": 3000,
            "shipmentDetails.chargeableWeight": 3100,
            "shipmentDetails.volumeWeight": 2900,
            "shipmentDetails.noOfContainers": 1,
            "shipmentDetails.originCity": "Shanghai",
            "shipmentDetails.originState": "Shanghai",
            "shipmentDetails.originCountry": "China",
            "shipmentDetails.destinationCity": "Mumbai",
            "shipmentDetails.destinationState": "Maharashtra",
            "shipmentDetails.destinationCountry": "India",
            "executionDate": "2026-01-01",
            "operational.clearanceStatus": "Pending",
            "operational.gateInStatus": "Pending",
            "operational.lastTrackingStatus": "Sailed",
            "operational.lastTrackingLocation": "Shanghai",
            "companyCode": "ZIPA",
        },
    ]
    return pd.DataFrame(rows)


def _dashboard_access_token(token_type: str = "access") -> str:
    return jwt.encode(
        {
            "sub": "dashboard-user",
            "role": "finance",
            "type": token_type,
            "exp": datetime.utcnow() + timedelta(minutes=30),
        },
        "dashboard-secret",
        algorithm="HS256",
    )


class DummyPayload:
    def __init__(self, **kwargs):
        self._payload = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self, exclude_none: bool = False):
        return dict(self._payload)


class ProductionRiskMainTests(unittest.TestCase):
    def test_descriptor_uses_local_champion(self):
        descriptor = describe_active_production_model(force_reload=True)
        self.assertEqual(descriptor["model_family"], "risk_main")
        self.assertEqual(descriptor["model_type"], "risk_main_xgb")
        self.assertEqual(descriptor["version"], "risk_main_xgb_20260319_094014")
        self.assertTrue(
            str(descriptor["model_path"]).endswith("models\\production\\risk_main_xgb_20260319_094014.pkl")
            or str(descriptor["model_path"]).endswith("models/production/risk_main_xgb_20260319_094014.pkl")
        )

    def test_manual_request_builder_accepts_nested_payload(self):
        payload = DummyPayload(
            customer={"customerId": "CUST-001", "customerName": "Acme Logistics"},
            shipmentDetails={"queryFor": "air", "accountType": "Export", "incoTerms": "FOB", "commodity": "Electronics"},
            receivables={"notDueAmount": 10000, "bucket0To15Amount": 5000},
            salesOwner="Riya Singh",
            currency="USD",
            taxableTotalAmountB=175000,
            grossAmount=180000,
            termsDays=30,
            invoiceDate="2026-02-10",
            paymentDateRaw="15/02, 20/03",
            paidAmount=90000,
            company="ZIPA",
        )
        frame = build_risk_main_manual_request_frame("air", payload)
        self.assertEqual(frame.loc[0, "customer.customerId"], "CUST-001")
        self.assertEqual(frame.loc[0, "shipmentDetails.queryFor"], "air")
        self.assertEqual(frame.loc[0, "shipmentDetails.incoTerms"], "FOB")
        self.assertEqual(int(frame.loc[0, "paymentInstallmentCount"]), 2)
        self.assertTrue(bool(frame.loc[0, "partialPaymentFlag"]))

    def test_scoring_frame_matches_local_artifact_contract(self):
        history = _history_frame()
        payload = DummyPayload(
            customerId="CUST-001",
            customerName="Acme Logistics",
            shipmentDetails={
                "queryFor": "air",
                "accountType": "Export",
                "incoTerms": "FOB",
                "commodity": "Electronics",
                "originCity": "Delhi",
                "originState": "Delhi",
                "originCountry": "India",
                "destinationCity": "Dubai",
                "destinationState": "Dubai",
                "destinationCountry": "UAE",
                "grossWeight": 1500,
                "chargeableWeight": 1580,
                "volumeWeight": 1490,
                "noOfContainers": 0,
            },
            invoiceDate="2026-02-15",
            dueDate="2026-03-16",
            salesOwner="Riya Singh",
            currency="USD",
            taxableTotalAmountB=210000,
            grossAmount=225000,
            paidAmount=0,
            tdsAmount=2000,
            ytdExposure=280000,
            company="ZIPA",
            customerAccountType="Credit",
            customerType="Existing",
            customerCategory="Key",
            customerOnboardDate="2024-01-15",
            notDueAmount=10000,
            bucket0To15Amount=8000,
            bucket16To30Amount=5000,
            bucket31To45Amount=3000,
            bucket46To60Amount=1000,
        )
        current = build_risk_main_manual_request_frame("air", payload)
        prod_frame = build_risk_main_scoring_frame(current, history_df=history)
        artifacts = load_production_artifacts(force_reload=True)
        features = artifacts["features"]
        model = artifacts["model"]
        preprocessor = artifacts["preprocessor"]

        self.assertEqual(len(features), 323)
        self.assertTrue(set(features).issubset(set(prod_frame.columns)))

        prod_prob = float(model.predict_proba(preprocessor.transform(prod_frame[features]))[:, 1][0])
        self.assertGreaterEqual(prod_prob, 0.0)
        self.assertLessEqual(prod_prob, 1.0)

    def test_api_supports_full_http_scoring_flow(self):
        history = _history_frame()
        nested_payload = {
            "customer": {"customerId": "CUST-001", "customerName": "Acme Logistics"},
            "shipmentDetails": {
                "queryFor": "air",
                "accountType": "Export",
                "incoTerms": "FOB",
                "commodity": "Electronics",
                "originCity": "Delhi",
                "originState": "Delhi",
                "originCountry": "India",
                "destinationCity": "Dubai",
                "destinationState": "Dubai",
                "destinationCountry": "UAE",
                "grossWeight": 1500,
                "chargeableWeight": 1580,
                "volumeWeight": 1490,
                "noOfContainers": 0,
            },
            "invoiceDate": "2026-02-15",
            "dueDate": "2026-03-16",
            "salesOwner": "Riya Singh",
            "currency": "USD",
            "taxableTotalAmountB": 210000,
            "grossAmount": 225000,
            "paidAmount": 0,
            "tdsAmount": 2000,
            "ytdExposure": 280000,
            "company": "ZIPA",
            "customerAccountType": "Credit",
            "customerType": "Existing",
            "customerCategory": "Key",
            "customerOnboardDate": "2024-01-15",
        }
        flat_payload = {
            "customerId": "CUST-001",
            "commodity": "Electronics",
            "taxableTotalAmountB": 210000,
            "incoTerms": "FOB",
            "currency": "USD",
            "weight_discrepancy": 80,
            "invoiceDate": "2026-02-15",
            "dueDate": "2026-03-16",
            "termsDays": 30,
            "grossAmount": 225000,
            "salesOwner": "Riya Singh",
            "documentType": "Invoice",
            "accountType": "Export",
            "company": "ZIPA",
        }

        mongo = MagicMock()
        mongo.command.return_value = {"ok": 1}

        with patch.dict(
            os.environ,
            {"API_KEY": "", "DASHBOARD_JWT_SECRET": "", "SECRET_KEY": ""},
            clear=False,
        ), patch.object(scoring_api._api_cache, "start", return_value=None), patch.object(
            scoring_api._api_cache, "stop", return_value=None
        ), patch.object(scoring_api._api_cache, "load_full_dataset", return_value=history), patch.object(
            scoring_api, "get_database", return_value=mongo
        ):
            with TestClient(scoring_api.app) as client:
                nested_response = client.post("/score/air", json=nested_payload)
                self.assertEqual(nested_response.status_code, 200)
                nested_result = nested_response.json()
                self.assertEqual(nested_result["model_version"], "risk_main_xgb_20260319_094014")
                self.assertIn("top_features", nested_result)

                legacy_response = client.post("/score/air", json=flat_payload)
                self.assertEqual(legacy_response.status_code, 200)
                legacy_result = legacy_response.json()
                self.assertEqual(legacy_result["model_version"], "risk_main_xgb_20260319_094014")
                self.assertIn("pd", legacy_result)

                score_all_response = client.get("/score-all/air?limit=1")
                self.assertEqual(score_all_response.status_code, 200)
                score_all_body = score_all_response.json()
                self.assertEqual(score_all_body["model_family"], "risk_main")
                self.assertEqual(score_all_body["count"], 1)
                self.assertEqual(score_all_body["total_available"], 2)
                self.assertTrue(score_all_body["pagination"]["has_more"])
                self.assertTrue(bool(score_all_body["pagination"]["next_cursor"]))
                self.assertEqual(score_all_body["snapshot_summary"]["rows"], 2)

                score_all_page_2 = client.get(
                    f"/score-all/air?limit=1&cursor={score_all_body['pagination']['next_cursor']}"
                )
                self.assertEqual(score_all_page_2.status_code, 200)
                score_all_page_2_body = score_all_page_2.json()
                self.assertEqual(score_all_page_2_body["count"], 1)
                self.assertFalse(score_all_page_2_body["pagination"]["has_more"])
                self.assertNotEqual(
                    score_all_body["records"][0]["invoice_key"],
                    score_all_page_2_body["records"][0]["invoice_key"],
                )

                live_customer_response = client.post("/score-customer/air", json={"customerId": "CUST-001", "limit": 2})
                self.assertEqual(live_customer_response.status_code, 200)
                live_customer_body = live_customer_response.json()
                self.assertEqual(live_customer_body["customer_summary"]["model_version"], "risk_main_xgb_20260319_094014")

                snapshot_id = score_all_body["pagination"]["snapshot_id"]
                customer_response = client.post(
                    "/score-customer/air",
                    json={"customerId": "CUST-001", "limit": 10, "snapshotId": snapshot_id},
                )
                self.assertEqual(customer_response.status_code, 200)
                customer_body = customer_response.json()
                self.assertEqual(customer_body["invoice_rows_scored"], 2)
                self.assertEqual(customer_body["customer_summary"]["model_version"], "risk_main_xgb_20260319_094014")

                health_response = client.get("/health")
                self.assertEqual(health_response.status_code, 200)
                health_result = health_response.json()
                self.assertEqual(health_result["production_model"]["model_family"], "risk_main")
                self.assertEqual(health_result["production_model"]["model_version"], "risk_main_xgb_20260319_094014")
                self.assertEqual(health_result["scored_snapshot"]["snapshot_id"], snapshot_id)
                self.assertEqual(health_result["mongo"], "ok")

                performance_response = client.get("/model-performance/air")
                self.assertEqual(performance_response.status_code, 200)
                performance_body = performance_response.json()
                self.assertEqual(performance_body["status"], "ok")
                self.assertEqual(performance_body["segment"], "air")
                self.assertEqual(performance_body["model"]["model_family"], "risk_main")
                self.assertIn("roc_auc", performance_body["live_metrics"])
                self.assertIn("comparison_metrics", performance_body)
                self.assertIn("confusion_matrix", performance_body)

    def test_customer_snapshot_id_validation(self):
        history = _history_frame()
        with patch.dict(
            os.environ,
            {"API_KEY": "", "DASHBOARD_JWT_SECRET": "", "SECRET_KEY": ""},
            clear=False,
        ), patch.object(scoring_api._api_cache, "start", return_value=None), patch.object(
            scoring_api._api_cache, "stop", return_value=None
        ), patch.object(scoring_api._api_cache, "load_full_dataset", return_value=history):
            with TestClient(scoring_api.app) as client:
                response = client.post(
                    "/score-customer/air",
                    json={"customerId": "CUST-001", "limit": 10, "snapshotId": "snapshot_missing"},
                )
                self.assertEqual(response.status_code, 422)

    def test_dashboard_access_token_allows_protected_routes(self):
        history = _history_frame()
        token = _dashboard_access_token()
        with patch.dict(os.environ, {"API_KEY": "", "DASHBOARD_JWT_SECRET": "dashboard-secret"}, clear=False), patch.object(scoring_api._api_cache, "start", return_value=None), patch.object(
            scoring_api._api_cache, "stop", return_value=None
        ), patch.object(scoring_api._api_cache, "load_full_dataset", return_value=history):
            with TestClient(scoring_api.app) as client:
                response = client.get(
                    "/score-all/air?limit=1",
                    headers={"Authorization": f"Bearer {token}"},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["count"], 1)

    def test_dashboard_token_required_when_secret_configured(self):
        history = _history_frame()
        with patch.dict(os.environ, {"API_KEY": "", "DASHBOARD_JWT_SECRET": "dashboard-secret"}, clear=False), patch.object(scoring_api._api_cache, "start", return_value=None), patch.object(
            scoring_api._api_cache, "stop", return_value=None
        ), patch.object(scoring_api._api_cache, "load_full_dataset", return_value=history):
            with TestClient(scoring_api.app) as client:
                response = client.get("/score-all/air?limit=1")
                self.assertEqual(response.status_code, 401)

    def test_openapi_exposes_pagination_and_snapshot_id(self):
        paths = scoring_api.app.openapi()["paths"]
        score_all_params = {
            item["name"]
            for item in paths["/score-all/{segment}"]["get"].get("parameters", [])
        }
        self.assertEqual(score_all_params, {"segment", "limit", "cursor", "refresh"})
        score_all_security = paths["/score-all/{segment}"]["get"].get("security", [])
        self.assertEqual(score_all_security, [{"RiskApiKey": []}, {"DashboardBearer": []}])
        model_performance_params = {
            item["name"]
            for item in paths["/model-performance/{segment}"]["get"].get("parameters", [])
        }
        self.assertEqual(model_performance_params, {"segment", "refresh"})
        model_performance_security = paths["/model-performance/{segment}"]["get"].get("security", [])
        self.assertEqual(model_performance_security, [{"RiskApiKey": []}, {"DashboardBearer": []}])

    def test_response_from_raw_avoids_fragmentation_warning(self):
        raw_df = pd.DataFrame(
            [
                {
                    "_id": "inv-001",
                    "invoiceNo": "INV-001",
                    "customer.customerId": "CUST-001",
                }
            ]
        )
        scored_df = pd.DataFrame(
            [
                {
                    **{f"feature_{idx}": float(idx) for idx in range(400)},
                    "pd": 0.42,
                    "score": 612,
                    "risk_band": "Medium Risk",
                }
            ]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", pd.errors.PerformanceWarning)
            merged = response_from_raw(raw_df, scored_df)

        perf_warnings = [item for item in caught if issubclass(item.category, pd.errors.PerformanceWarning)]
        self.assertEqual(perf_warnings, [])
        self.assertEqual(len(merged.columns), len(raw_df.columns) + len(scored_df.columns))
        self.assertEqual(merged.loc[0, "invoiceNo"], "INV-001")
        self.assertEqual(float(merged.loc[0, "pd"]), 0.42)

    def test_score_returns_500_when_model_file_missing(self):
        history = _history_frame()
        with patch.dict(
            os.environ,
            {"API_KEY": "", "DASHBOARD_JWT_SECRET": "", "SECRET_KEY": ""},
            clear=False,
        ), patch.object(scoring_api, "_startup_checks", return_value=None), patch.object(
            scoring_api._api_cache, "start", return_value=None
        ), patch.object(scoring_api._api_cache, "stop", return_value=None), patch.object(
            scoring_api._api_cache, "load_full_dataset", return_value=history
        ), patch("Riskpredictionmodel.scoring.model.load_production_artifacts", side_effect=FileNotFoundError("models/production/missing.pkl")):
            with TestClient(scoring_api.app) as client:
                response = client.post(
                    "/score/air",
                    json={
                        "customerId": "CUST-001",
                        "taxableTotalAmountB": 1000,
                        "invoiceDate": "2026-02-15",
                        "dueDate": "2026-03-16",
                    },
                )
                self.assertEqual(response.status_code, 500)
                self.assertEqual(response.json()["detail"], "Scoring failed. Check server logs.")
                self.assertNotIn("missing.pkl", response.json()["detail"])

    def test_score_all_respects_page_max(self):
        history = _history_frame()
        with patch.dict(
            os.environ,
            {"API_KEY": "", "DASHBOARD_JWT_SECRET": "", "SECRET_KEY": ""},
            clear=False,
        ), patch.object(scoring_api._api_cache, "start", return_value=None), patch.object(
            scoring_api._api_cache, "stop", return_value=None
        ), patch.object(scoring_api._api_cache, "load_full_dataset", return_value=history):
            with TestClient(scoring_api.app) as client:
                response = client.get("/score-all/all?limit=999999")
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertLessEqual(body["count"], scoring_api.SETTINGS.score_all_page_max)
                self.assertEqual(body["limit"], scoring_api.SETTINGS.score_all_page_max)

    def test_health_returns_503_when_mongo_unreachable(self):
        mongo = MagicMock()
        mongo.command.side_effect = ServerSelectionTimeoutError("mongo down")
        with patch.dict(
            os.environ,
            {"API_KEY": "", "DASHBOARD_JWT_SECRET": "", "SECRET_KEY": ""},
            clear=False,
        ), patch.object(scoring_api._api_cache, "start", return_value=None), patch.object(
            scoring_api._api_cache, "stop", return_value=None
        ), patch.object(scoring_api, "get_database", return_value=mongo):
            with TestClient(scoring_api.app) as client:
                response = client.get("/health")
                self.assertEqual(response.status_code, 503)
                body = response.json()
                self.assertEqual(body["status"], "degraded")
                self.assertEqual(body["mongo"], "unreachable")

    def test_score_request_rejects_negative_amount(self):
        with patch.dict(
            os.environ,
            {"API_KEY": "", "DASHBOARD_JWT_SECRET": "", "SECRET_KEY": ""},
            clear=False,
        ), patch.object(scoring_api._api_cache, "start", return_value=None), patch.object(
            scoring_api._api_cache, "stop", return_value=None
        ):
            with TestClient(scoring_api.app) as client:
                response = client.post(
                    "/score/air",
                    json={"customerId": "CUST-001", "taxableTotalAmountB": -1},
                )
                self.assertEqual(response.status_code, 422)

    def test_customer_score_rejects_empty_customer_id(self):
        with patch.dict(
            os.environ,
            {"API_KEY": "", "DASHBOARD_JWT_SECRET": "", "SECRET_KEY": ""},
            clear=False,
        ), patch.object(scoring_api._api_cache, "start", return_value=None), patch.object(
            scoring_api._api_cache, "stop", return_value=None
        ):
            with TestClient(scoring_api.app) as client:
                response = client.post("/score-customer/air", json={"customerId": "", "limit": 10})
                self.assertEqual(response.status_code, 400)

    def test_build_customer_history_aggregates_empty_input(self):
        result = build_customer_history_aggregates(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_model_performance_degrades_when_snapshot_unavailable(self):
        with patch.dict(
            os.environ,
            {"API_KEY": "", "DASHBOARD_JWT_SECRET": "", "SECRET_KEY": ""},
            clear=False,
        ), patch.object(scoring_api, "_startup_checks", return_value=None), patch.object(
            scoring_api._api_cache, "start", return_value=None
        ), patch.object(scoring_api._api_cache, "stop", return_value=None), patch.object(
            scoring_api._api_cache, "get_scored_segment_frame", side_effect=ServerSelectionTimeoutError("mongo down")
        ):
            with TestClient(scoring_api.app) as client:
                response = client.get("/model-performance/all")
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual(body["status"], "degraded")
                self.assertEqual(body["live_status"], "unavailable")
                self.assertEqual(body["live_metrics"]["rows"], 0)
                self.assertIn("registry_metrics", body)


if __name__ == "__main__":
    unittest.main()

