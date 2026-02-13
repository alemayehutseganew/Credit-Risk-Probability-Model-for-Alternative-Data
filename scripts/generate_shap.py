"""
Generate SHAP explainability plots for the trained credit risk model.

Usage:
    python scripts/generate_shap.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "credit-risk-model-best-model" / "model.pkl"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_SCHEMA_PATH = PROCESSED_DIR / "feature_schema.json"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


def _load_feature_columns() -> list[str]:
    if not FEATURE_SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Missing feature schema at {FEATURE_SCHEMA_PATH}")
    schema = json.loads(FEATURE_SCHEMA_PATH.read_text())
    features = schema.get("features", [])
    if not features:
        raise ValueError("Feature schema is empty")
    return list(features)


def _load_training_sample(feature_columns: list[str]) -> pd.DataFrame:
    data_path = PROCESSED_DIR / "X_train.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing training data at {data_path}")
    df = pd.read_csv(data_path)
    return df[feature_columns]


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")

    feature_columns = _load_feature_columns()
    X = _load_training_sample(feature_columns)

    # Sample for faster plotting and stable runtime
    X_sample = X.sample(min(200, len(X)), random_state=42)

    logger.info("Loading model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    logger.info("Computing SHAP values")
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    _ensure_output_dir()

    summary_path = OUTPUT_DIR / "shap_summary.png"
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close()

    waterfall_path = OUTPUT_DIR / "shap_waterfall.png"
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=160, bbox_inches="tight")
    plt.close()

    logger.info("Saved SHAP summary to %s", summary_path)
    logger.info("Saved SHAP waterfall to %s", waterfall_path)


if __name__ == "__main__":
    main()
