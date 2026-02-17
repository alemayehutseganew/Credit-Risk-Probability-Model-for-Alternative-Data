"""Integration tests for FastAPI endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi.testclient import TestClient

from src.api import main as api_main


class DummyModel:
    """Simple model stub with deterministic probabilities."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, X) -> np.ndarray:
        probs = np.full(len(X), self.probability, dtype=float)
        return np.column_stack([1 - probs, probs])


def _build_client(monkeypatch) -> TestClient:
    @asynccontextmanager
    async def _lifespan_override(app):
        yield

    monkeypatch.setattr(api_main.app.router, "lifespan_context", _lifespan_override)

    api_main.model = DummyModel(0.2)
    api_main.feature_columns = [
        "Recency",
        "Frequency",
        "Monetary",
        "total_amount",
        "avg_amount",
        "std_amount",
        "min_amount",
        "max_amount",
        "transaction_count",
        "total_value",
        "avg_value",
        "std_value",
        "min_value",
        "max_value",
        "debit_ratio",
        "credit_ratio",
        "transaction_hour_mean",
        "transaction_day_mean",
        "transaction_month_mean",
        "transaction_year_mean",
        "transaction_dayofweek_mean",
        "weekend_transaction_ratio",
        "primary_channel_woe",
        "primary_category_woe",
        "primary_currency_woe",
        "primary_pricing_woe",
    ]
    api_main.woe_mappings = {
        "primary_channel": {"MOBILE": 0.1},
        "primary_category": {"AIRTIME": -0.2},
        "primary_currency": {"UGX": 0.05},
        "primary_pricing": {"PRICING_STRATEGY_1": 0.3},
    }
    api_main.preprocessor = None

    return TestClient(api_main.app)


def _payload() -> dict:
    return {
        "customer_id": "CUST_1",
        "Recency": 10,
        "Frequency": 5,
        "Monetary": 500,
        "total_amount": 1000,
        "avg_amount": 100,
        "std_amount": 10,
        "min_amount": 5,
        "max_amount": 200,
        "transaction_count": 5,
        "total_value": 900,
        "avg_value": 90,
        "std_value": 9,
        "min_value": 3,
        "max_value": 220,
        "debit_ratio": 0.7,
        "credit_ratio": 0.3,
        "transaction_hour_mean": 10,
        "transaction_day_mean": 12,
        "transaction_month_mean": 6,
        "transaction_year_mean": 2019,
        "transaction_dayofweek_mean": 2,
        "weekend_transaction_ratio": 0.1,
        "primary_channel": "MOBILE",
        "primary_category": "AIRTIME",
        "primary_currency": "UGX",
        "primary_pricing": "PRICING_STRATEGY_1",
    }


def test_health_check(monkeypatch):
    client = _build_client(monkeypatch)

    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_version"] == "production"


def test_predict_endpoint(monkeypatch):
    client = _build_client(monkeypatch)

    response = client.post("/predict", json=_payload())
    assert response.status_code == 200

    payload = response.json()
    assert 0.0 <= payload["risk_probability"] <= 1.0
    assert payload["risk_category"] == "low_risk"
    assert 300 <= payload["credit_score"] <= 850
    assert api_main.SCORING_CONFIG.min_amount <= payload["recommended_amount"] <= api_main.SCORING_CONFIG.max_amount
    assert api_main.SCORING_CONFIG.min_months <= payload["recommended_duration_months"] <= api_main.SCORING_CONFIG.max_months


def test_predict_batch_endpoint(monkeypatch):
    client = _build_client(monkeypatch)

    response = client.post("/predict-batch", json=[_payload(), _payload()])
    assert response.status_code == 200

    payload = response.json()
    assert payload["count"] == 2
    assert len(payload["predictions"]) == 2
