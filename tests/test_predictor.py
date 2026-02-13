"""Tests for prediction utilities and API feature frame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.predict import CreditRiskPredictor
from src.api import main as api_main
from src.api.pydantic_models import CustomerFeatures


class DummyModel:
    """Simple model stub with deterministic probabilities."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.full(len(X), self.probability, dtype=float)
        return np.column_stack([1 - probs, probs])


def _make_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Recency": [5.0],
            "Frequency": [20.0],
            "Monetary": [5000.0],
            "total_amount": [1000.0],
            "avg_amount": [50.0],
            "std_amount": [10.0],
            "min_amount": [5.0],
            "max_amount": [200.0],
            "transaction_count": [20],
            "total_value": [1200.0],
            "avg_value": [60.0],
            "std_value": [12.0],
            "min_value": [2.0],
            "max_value": [210.0],
            "debit_ratio": [0.8],
            "credit_ratio": [0.2],
            "transaction_hour_mean": [12.0],
            "transaction_day_mean": [15.0],
            "transaction_month_mean": [6.0],
            "transaction_year_mean": [2019.0],
            "transaction_dayofweek_mean": [3.0],
            "weekend_transaction_ratio": [0.1],
        }
    )


def test_predict_credit_score_monotonic():
    predictor = CreditRiskPredictor()
    predictor.model = DummyModel(0.2)

    scores = predictor.predict_credit_score(_make_feature_frame())
    assert scores[0] == 740


def test_predict_loan_amount_bounds():
    predictor = CreditRiskPredictor()
    predictor.model = DummyModel(0.0)

    amounts = predictor.predict_loan_amount(_make_feature_frame())
    assert amounts[0] == predictor.scoring_config.max_amount


def test_predict_loan_duration_bounds():
    predictor = CreditRiskPredictor()
    predictor.model = DummyModel(1.0)

    durations = predictor.predict_loan_duration(_make_feature_frame())
    assert durations[0] == predictor.scoring_config.min_months


def test_predict_risk_category_threshold_override():
    predictor = CreditRiskPredictor()
    predictor.model = DummyModel(0.49)

    categories = predictor.predict_risk_category(_make_feature_frame(), threshold=0.4)
    assert categories[0] == "high_risk"


def test_build_feature_frame_woe_mapping(monkeypatch):
    feature_columns = [
        "Recency",
        "Frequency",
        "Monetary",
        "primary_channel_woe",
        "primary_category_woe",
        "primary_currency_woe",
        "primary_pricing_woe",
    ]
    woe_mappings = {
        "primary_channel": {"MOBILE": 0.3},
        "primary_category": {"AIRTIME": -0.2},
        "primary_currency": {"UGX": 0.1},
        "primary_pricing": {"PRICING_STRATEGY_1": 0.4},
    }

    monkeypatch.setattr(api_main, "feature_columns", feature_columns)
    monkeypatch.setattr(api_main, "woe_mappings", woe_mappings)
    monkeypatch.setattr(api_main, "preprocessor", None)

    payload = CustomerFeatures(
        customer_id="CUST_1",
        Recency=10,
        Frequency=5,
        Monetary=500,
        total_amount=1000,
        avg_amount=100,
        std_amount=10,
        min_amount=5,
        max_amount=200,
        transaction_count=5,
        total_value=900,
        avg_value=90,
        std_value=9,
        min_value=3,
        max_value=220,
        debit_ratio=0.7,
        credit_ratio=0.3,
        transaction_hour_mean=10,
        transaction_day_mean=12,
        transaction_month_mean=6,
        transaction_year_mean=2019,
        transaction_dayofweek_mean=2,
        weekend_transaction_ratio=0.1,
        primary_channel="MOBILE",
        primary_category="AIRTIME",
        primary_currency="UGX",
        primary_pricing="PRICING_STRATEGY_1",
    )

    customer_id, frame = api_main.build_feature_frame(payload)

    assert customer_id == "CUST_1"
    assert frame.shape == (1, len(feature_columns))
    assert frame.loc[0, "primary_channel_woe"] == 0.3
    assert frame.loc[0, "primary_category_woe"] == -0.2
